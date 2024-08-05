use crate::{CommunityID, IterationStep};

use super::{displs_from_counts, CommunityState, LouvainGraph, LouvainMessage, MessageRouter};

use log::{debug, error, info, trace};

use indexmap::{indexset, IndexMap, IndexSet};
use petgraph::{csr::Csr, visit::EdgeRef, Directed, IntoWeightedEdge};

use anyhow::anyhow;

use mpi::{
    collective::SystemOperation,
    datatype::{Partition, PartitionMut},
    topology::{Process, SimpleCommunicator},
    traits::{Communicator, CommunicatorCollectives, Equivalence, PartitionedBuffer},
    Count, Rank,
};

/// Newtype to represent vertex ids
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Equivalence, PartialOrd, Ord, Hash)]
pub struct VertexID(pub usize);

impl From<usize> for VertexID {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Debug, Equivalence, Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Edge(pub VertexID, pub VertexID, pub f64);

impl From<(usize, usize, f64)> for Edge {
    fn from(value: (usize, usize, f64)) -> Self {
        Self(value.0.into(), value.1.into(), value.2)
    }
}

impl IntoWeightedEdge<f64> for Edge {
    type NodeId = usize;
    fn into_weighted_edge(self) -> (Self::NodeId, Self::NodeId, f64) {
        (self.0 .0, self.1 .0, self.2)
    }
}

/// Stores information about our distributed setup
pub struct DistributedInfo<'a> {
    world: &'a SimpleCommunicator,
    pub rank: usize,
    /// size of MPI_COMM_WORLD
    pub size: usize,
    /// rows of the global adj matrix
    pub owned_rows: (usize, usize),
    /// idx = rank & element rows for that rank exclusive rows.0..rows.1
    row_ownership: Vec<(usize, usize)>,

    global_vcount: usize,
    global_ecount: usize,

    most_vertices_owned_by_a_process: Option<usize>,
}

impl std::fmt::Debug for DistributedInfo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedInfo")
            .field("rank", &self.rank)
            .field("size", &self.size)
            .field("owned_rows", &self.owned_rows)
            .finish_non_exhaustive()
    }
}

impl<'a> DistributedInfo<'a> {
    /// Given the size of the MPI topology, and the global number of vertices, compute the row
    /// ownership of every rank
    pub fn compute_vertex_distribution(size: usize, global_vcount: usize) -> Vec<(usize, usize)> {
        let mut row_ownership = Vec::with_capacity(size);
        row_ownership.resize(size, (0, 0));

        let vertices_per_proc = global_vcount / size;
        let leftover_verts = global_vcount % size;

        for (rank, rows) in row_ownership.iter_mut().enumerate() {
            let leftover = if rank < leftover_verts { 1 } else { 0 };
            rows.0 = rank * vertices_per_proc + std::cmp::min(rank, leftover_verts);
            rows.1 = rows.0 + vertices_per_proc + leftover;
        }

        row_ownership
    }

    /// Return the global number of vertices in the graph
    pub fn global_vcount(&self) -> usize {
        self.global_vcount
    }

    /// Return the global number of vertices in the graph
    pub fn global_ecount(&self) -> usize {
        self.global_ecount
    }

    // private helper for keeping track of local state
    pub fn local_vcount(&self) -> usize {
        self.owned_rows.1 - self.owned_rows.0
    }

    fn max_vertices_owned_by_any_proc(&self) -> usize {
        self.most_vertices_owned_by_a_process.unwrap_or_else(|| {
            self.row_ownership
                .iter()
                .map(|(start, end)| end - start)
                .max()
                .unwrap_or(0)
        })
    }

    /// Given a vtx or a community, determine who owns it
    /// by searching the row_ownership vector
    fn owner_of(&self, vtx: usize) -> Rank {
        assert!(self.local_vcount() != 0);

        // this should get us close enough
        let mut guess = vtx / (self.local_vcount() + 1);
        loop {
            let rows = self.row_ownership[guess];
            if vtx < rows.0 {
                guess -= 1;
            } else if vtx >= rows.1 {
                guess += 1;
            } else {
                return guess as Rank;
            }
        }
    }

    pub fn init(world: &'a SimpleCommunicator, global_vcount: usize) -> Self {
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        // compute row ownership
        let row_ownership = DistributedInfo::compute_vertex_distribution(size, global_vcount);
        let owned_rows = row_ownership[rank];

        Self {
            world,
            rank,
            size,
            owned_rows,
            row_ownership,
            global_vcount,
            // this field needs to be set later with the proper global edge count
            // this can only be done after all the edges have been processed
            global_ecount: 0,
            most_vertices_owned_by_a_process: None,
        }
    }
}

/// Simple wrapper over petgraph CSR to represent a Distributed Graph
/// This graph is Directed
#[derive(Debug)]
pub struct DistributedGraph<'a> {
    pub inner: Csr<VertexID, f64, Directed, usize>,

    local_vcount: usize,
    local_ecount: usize,

    // info about global graph
    pub info: DistributedInfo<'a>,
}

impl<'a> DistributedGraph<'a> {
    /// Builds the graph by communicating with MPI Processes
    pub fn from_distributed(edges: &[Edge], world: &'a SimpleCommunicator) -> anyhow::Result<Self> {
        info!("Building DistributedGraph from {} edges", edges.len());
        let global_vcount = Self::compute_global_graph_info(edges, world)?;
        info!("Computed global vertex count: {}", global_vcount);

        let mut info = DistributedInfo::init(world, global_vcount);
        debug!("Initialized DistributedInfo: {:?}", info);

        let sorted_edges = Self::distribute_and_gather_local_edges(edges, &info)?;
        info!(
            "Distributed and gathered {} local edges",
            sorted_edges.len()
        );

        // global ecount
        let mut global_ecount = 0;
        world.all_reduce_into(
            &sorted_edges.len(),
            &mut global_ecount,
            SystemOperation::sum(),
        );
        info.global_ecount = global_ecount;
        info!("Computed global edge count: {}", global_ecount);

        let csr = Csr::from_sorted_edges(&sorted_edges).map_err(|_| {
            let err = anyhow!("Failed to build CSR because edges are not sorted");
            error!("{}", err);
            err
        })?;
        debug!(
            "Built CSR graph with {} nodes and {} edges",
            csr.node_count(),
            csr.edge_count()
        );

        Ok(Self {
            local_vcount: info.local_vcount(),
            local_ecount: csr.edge_count(),
            inner: csr,
            info,
        })
    }

    /// number of edges between our owned nodes and all other nodes in the graph
    pub fn edge_count(&self) -> usize {
        trace!("Returning local edge count: {}", self.local_ecount);
        self.local_ecount
    }

    // Helper functions

    /// given the edge list, communicate with all ranks and determine the global vcount
    fn compute_global_graph_info(
        edges: &[Edge],
        world: &SimpleCommunicator,
    ) -> anyhow::Result<usize> {
        debug!("Computing global graph info");
        let local_max_vtx = edges
            .iter()
            .map(|e| std::cmp::max(e.0, e.1))
            .max()
            .unwrap_or_default();
        debug!("Local max vertex: {:?}", local_max_vtx);

        // communicate max vtx and determine how many vertices are in the total distributed graph
        let mut global_max_vtx: usize = usize::MAX;
        world.all_reduce_into(
            &local_max_vtx.0,
            &mut global_max_vtx,
            SystemOperation::max(),
        );
        debug!("Global max vertex after reduction: {}", global_max_vtx);

        // error checking: return global_max_vtx + 1 if MPI call was successful
        // or an error if it was not set properly
        if global_max_vtx == 0 {
            info!("Graph is empty");
            Ok(0)
        } else if global_max_vtx == usize::MAX {
            let err = anyhow!("Failed to determine global graph info");
            error!("{}", err);
            Err(err)
        } else {
            let result = global_max_vtx + 1;
            info!("Computed global vertex count: {}", result);
            Ok(result)
        }
    }

    /// Takes a list of edges, and creates a vector of Edge structs partitioned by
    /// rank owner of that edge, partitioning determined by counts, and displs vectors
    fn partition_edges_by_rank(
        info: &DistributedInfo,
        edges: &[Edge],
    ) -> (Vec<Edge>, Vec<Count>, Vec<Count>) {
        debug!("Partitioning {} edges by rank", edges.len());
        // organize edges by rank
        let mut rank_edges = IndexMap::<usize, Vec<Edge>>::with_capacity(info.size);

        edges.iter().for_each(|edge| {
            let rank1 = info.owner_of(edge.0 .0) as usize;
            let rank2 = info.owner_of(edge.1 .0) as usize;

            // compute both directions of the edge
            rank_edges.entry(rank1).or_default().push(*edge);
            trace!("Assigned edge {:?} to rank {}", edge, rank1);

            // self loops are added once
            if edge.0 != edge.1 {
                let reverse_edge = Edge(edge.1, edge.0, edge.2);
                rank_edges.entry(rank2).or_default().push(reverse_edge);
                trace!("Assigned reverse edge {:?} to rank {}", reverse_edge, rank2);
            }
        });

        let mut counts = vec![0; info.size];
        let mut displs = vec![0; info.size];
        let mut total_edges = 0;

        for rank in 0..info.size {
            let count = rank_edges
                .get(&rank)
                .map_or(0, |edges| edges.len() as Count);

            counts[rank] = count;
            displs[rank] = total_edges;
            total_edges += count;
            trace!(
                "Rank {}: count = {}, displacement = {}",
                rank,
                count,
                displs[rank]
            );
        }

        // be sure that we flatten by rank
        rank_edges.sort_unstable_keys();

        let buf = rank_edges
            .into_iter()
            .flat_map(|(_, mut edges)| {
                edges.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
                edges
            })
            .collect();

        debug!(
            "Partitioned edges: total = {}, counts = {:?}",
            total_edges, counts
        );
        (buf, counts, displs)
    }

    /// Communicate with all ranks and determine how many edges are being sent to use
    fn gather_incoming_edge_counts<M: PartitionedBuffer>(
        info: &DistributedInfo,
        msg: &M,
    ) -> (Vec<Count>, Vec<Count>) {
        debug!("Gathering incoming edge counts");
        // initialize buffer and send counts to all processes
        let mut recv_counts = vec![0; info.size];
        info.world.all_to_all_into(msg.counts(), &mut recv_counts);
        debug!("Received edge counts: {:?}", recv_counts);

        let recv_displs = displs_from_counts(&recv_counts);
        debug!("Calculated displacements: {:?}", recv_displs);
        (recv_counts, recv_displs)
    }

    /// Build the send and receive buffer based on the edge list and our Distributed setup
    /// and perform communication, returning a complete sorted list of edges belonging to this process
    fn distribute_and_gather_local_edges(
        edges: &[Edge],
        info: &DistributedInfo,
    ) -> anyhow::Result<Vec<Edge>> {
        info!("Distributing and gathering local edges");
        // build send buffer
        let (send_buf, send_counts, send_displs) = Self::partition_edges_by_rank(info, edges);
        debug!("Send buffer prepared: {} edges", send_buf.len());

        let send = Partition::new(&send_buf, send_counts, send_displs);

        // build recv buffer
        let (recv_counts, recv_displs) = Self::gather_incoming_edge_counts(info, &send);
        let recv_total: i32 = recv_counts.iter().sum();
        debug!("Expecting to receive {} edges", recv_total);
        let mut recv_buf = vec![Edge::default(); recv_total.try_into()?];
        let mut recv = PartitionMut::new(&mut recv_buf, recv_counts, recv_displs);

        // communicate
        info!("Performing all_to_all_varcount communication");
        info.world.all_to_all_varcount_into(&send, &mut recv);
        debug!("Completed all_to_all_varcount communication");

        // sort edges and create graph
        recv_buf.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        info!("Sorted {} received edges", recv_buf.len());

        // return data buffer
        Ok(recv_buf)
    }
}

impl<'a> LouvainGraph for DistributedGraph<'a> {
    fn local_vertex_count(&self) -> usize {
        trace!("Returning local vertex count: {}", self.local_vcount);
        self.local_vcount
    }

    fn global_vertex_count(&self) -> usize {
        trace!(
            "Returning global vertex count: {}",
            self.info.global_vcount()
        );
        self.info.global_vcount()
    }

    fn global_edge_count(&self) -> usize {
        trace!("Returning global edge count: {}", self.info.global_ecount());
        self.info.global_ecount()
    }

    fn weighted_degree(&self, vertex: usize) -> f64 {
        let degree = self.inner.edges(vertex).map(|e| e.weight()).sum();
        trace!("Computed weighted degree for vertex {}: {}", vertex, degree);
        degree
    }

    fn neighbors(&self, vertex: VertexID) -> impl Iterator<Item = (VertexID, f64)> {
        trace!("Getting neighbors for vertex {:?}", vertex);
        self.inner
            .edges(vertex.0)
            .map(|e| (e.target().into(), *e.weight()))
    }

    fn vertices(&self) -> impl Iterator<Item = VertexID> {
        trace!("Getting vertices owned by this graph");
        (self.info.owned_rows.0..self.info.owned_rows.1)
            .map(|v| VertexID(v))
            .into_iter()
    }

    // TODO: Random order of vertices
    fn vertex_feed(&self) -> impl Iterator<Item = IterationStep> {
        // we need this many items of IterationStep::SyncOnly at the end
        let difference = self.info.max_vertices_owned_by_any_proc() - self.local_vertex_count();

        self.vertices()
            .map(|v| IterationStep::Process(v))
            .chain((0..difference).map(|_| IterationStep::SyncOnly))
    }

    fn is_local_vertex(&self, vertex: usize) -> bool {
        let is_local = self.info.rank == self.info.owner_of(vertex) as usize;
        trace!("Checking if vertex {} is local: {}", vertex, is_local);
        is_local
    }

    fn owner_of_vertex(&self, vtx: VertexID) -> Rank {
        let owner = self.info.owner_of(vtx.0);
        trace!("Owner of vertex {:?}: Rank {}", vtx, owner);
        owner
    }

    fn owner_of_community(&self, community: CommunityID) -> Rank {
        let owner = self.info.owner_of(community.0);
        trace!("Owner of community {:?}: Rank {}", community, owner);
        owner
    }
}

impl MessageRouter for DistributedGraph<'_> {
    fn route<S: CommunityState>(&self, message: LouvainMessage, state: &S) -> IndexSet<Rank> {
        match message {
            LouvainMessage::CommunityInfo(update) => {
                // update all ranks with a vtx in this community
                let mut set: IndexSet<Rank> = IndexSet::with_capacity(self.size() / 2);

                for vtx in state.get_community_members(&update.id) {
                    set.insert(self.owner_of_vertex(*vtx));
                }
                set
            }
            // only send to the owner of the target community
            LouvainMessage::VertexMovement(movement) => {
                indexset! {self.owner_of_community(movement.community_id)}
            }
            LouvainMessage::NeighborInfo(info) => {
                let mut set = IndexSet::with_capacity(self.size() / 2);

                for (vtx, _) in self.neighbors(info.vertex_id) {
                    set.insert(self.owner_of_vertex(vtx));
                }

                set
            }
        }
    }

    fn process_at_rank(&self, rank: Rank) -> Process {
        trace!("Getting process for rank {}", rank);
        self.info.world.process_at_rank(rank)
    }

    fn size(&self) -> usize {
        trace!("Getting communicator size: {}", self.info.size);
        self.info.size
    }

    fn this_process(&self) -> Process {
        trace!("Getting this process");
        self.info.world.this_process()
    }

    fn distribute_counts(&self, buf: &[Count], recv_counts: &mut [Count]) -> anyhow::Result<()> {
        debug!("Distributing counts, buffer length: {}", buf.len());
        if buf.len() != self.size() {
            let err = anyhow!("Buffer length must match topology");
            error!("{}", err);
            return Err(err);
        }

        self.info
            .world
            .all_reduce_into(buf, recv_counts, SystemOperation::sum());
        debug!("Distributed counts: {:?}", recv_counts);

        Ok(())
    }

    fn global_modularity(&self, partial: f64) -> f64 {
        debug!("Computing global modularity, partial: {}", partial);
        let mut sum = f64::default();
        self.info
            .world
            .all_reduce_into(&partial, &mut sum, SystemOperation::sum());
        info!("Computed global modularity: {}", sum);
        sum
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockInfoBuilder {
        rank: Option<usize>,
        size: Option<usize>,
        owned_rows: Option<(usize, usize)>,
        row_ownership: Option<Vec<(usize, usize)>>,
        global_vcount: Option<usize>,
        global_ecount: Option<usize>,
    }

    impl MockInfoBuilder {
        fn new() -> Self {
            Self::default()
        }

        fn rank(mut self, rank: usize) -> Self {
            self.rank = Some(rank);
            self
        }

        fn size(mut self, size: usize) -> Self {
            self.size = Some(size);
            self
        }

        fn owned_rows(mut self, owned_rows: (usize, usize)) -> Self {
            self.owned_rows = Some(owned_rows);
            self
        }

        fn row_ownership(mut self, row_ownership: Vec<(usize, usize)>) -> Self {
            self.row_ownership = Some(row_ownership);
            self
        }

        fn global_vcount(mut self, global_vcount: usize) -> Self {
            self.global_vcount = Some(global_vcount);
            self
        }

        fn global_ecount(mut self, global_ecount: usize) -> Self {
            self.global_ecount = Some(global_ecount);
            self
        }

        fn build<'a>(self) -> anyhow::Result<DistributedInfo<'a>> {
            let rank = self.rank.unwrap_or(0);
            let size = self
                .size
                .ok_or_else(|| anyhow::anyhow!("size field not set"))?;
            let global_vcount = self
                .global_vcount
                .ok_or_else(|| anyhow::anyhow!("global_vcount field not set"))?;
            let global_ecount = self.global_ecount.unwrap_or(0);

            let row_ownership = self.row_ownership.unwrap_or_else(|| {
                DistributedInfo::compute_vertex_distribution(size, global_vcount)
            });

            let owned_rows = self.owned_rows.unwrap_or_else(|| row_ownership[rank]);

            let world: &'a SimpleCommunicator =
                unsafe { std::mem::MaybeUninit::zeroed().assume_init() };

            Ok(DistributedInfo {
                rank,
                size,
                world,
                owned_rows,
                row_ownership,
                global_vcount,
                global_ecount,
                most_vertices_owned_by_a_process: None,
            })
        }
    }

    #[test]
    fn test_even_distribution() {
        let size = 4;
        let global_vcount = 100;
        let distribution = DistributedInfo::compute_vertex_distribution(size, global_vcount);
        assert_eq!(distribution, vec![(0, 25), (25, 50), (50, 75), (75, 100)]);
    }

    #[test]
    fn test_uneven_distribution() {
        let size = 3;
        let global_vcount = 10;
        let distribution = DistributedInfo::compute_vertex_distribution(size, global_vcount);
        assert_eq!(distribution, vec![(0, 4), (4, 7), (7, 10)]);
    }

    #[test]
    fn test_zero_vertices() {
        let size = 5;
        let global_vcount = 0;
        let distribution = DistributedInfo::compute_vertex_distribution(size, global_vcount);
        assert_eq!(distribution, vec![(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]);
    }

    #[test]
    fn test_zero_size() {
        let size = 0;
        let global_vcount = 10;
        let distribution = DistributedInfo::compute_vertex_distribution(size, global_vcount);
        assert_eq!(distribution, vec![]);
    }

    #[test]
    fn test_rank_ownership() {
        // very scary
        let world: SimpleCommunicator = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        let rank = 0;
        let size = 4;

        // assume 4 processes
        let dist = DistributedInfo::compute_vertex_distribution(4, 40);
        let dinfo = DistributedInfo {
            world: &world,
            rank,
            size,
            owned_rows: dist[rank],
            row_ownership: dist,
            global_vcount: 40,
            global_ecount: 69,
            most_vertices_owned_by_a_process: None,
        };

        let mut expected_owner = 0;
        assert_eq!(expected_owner, dinfo.owner_of(0) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(5) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(9) as usize);

        expected_owner = 2;
        assert_eq!(expected_owner, dinfo.owner_of(20) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(25) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(29) as usize);

        // assume 4 processes, and an uneven distribution
        // distribution = (0, 3), (3, 6), (6, 8), (8, 10)]
        let dist = DistributedInfo::compute_vertex_distribution(4, 10);
        let dinfo = DistributedInfo {
            world: &world,
            rank,
            size,
            owned_rows: dist[rank],
            row_ownership: dist,
            global_vcount: 10,
            global_ecount: 5,
            most_vertices_owned_by_a_process: None,
        };

        let mut expected_owner = 0;
        assert_eq!(expected_owner, dinfo.owner_of(0) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(1) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(2) as usize);

        expected_owner = 1;
        assert_eq!(expected_owner, dinfo.owner_of(3) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(4) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(5) as usize);

        expected_owner = 2;
        assert_eq!(expected_owner, dinfo.owner_of(6) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(7) as usize);

        expected_owner = 3;
        assert_eq!(expected_owner, dinfo.owner_of(8) as usize);
        assert_eq!(expected_owner, dinfo.owner_of(9) as usize);
    }

    #[test]
    fn test_partition_single_edge() {
        // Assumption: A single edge should be correctly assigned to its owner rank
        let info = MockInfoBuilder::new()
            .size(2)
            .global_vcount(2)
            .build()
            .unwrap();
        let edges = vec![Edge::from((0, 1, 1.0))];

        let (buf, counts, displs) = DistributedGraph::partition_edges_by_rank(&info, &edges);

        assert_eq!(buf, vec![Edge::from((0, 1, 1.0)), Edge::from((1, 0, 1.0))]);
        assert_eq!(counts, vec![1, 1]);
        assert_eq!(displs, vec![0, 1]);
    }

    #[test]
    fn test_partition_multiple_edges_same_rank() {
        // Assumption: Multiple edges belonging to the same rank should be grouped together
        let info = MockInfoBuilder::new()
            .size(2)
            .global_vcount(8)
            .build()
            .unwrap();
        let edges = vec![Edge::from((0, 1, 1.0)), Edge::from((2, 3, 2.0))];

        let (buf, counts, displs) = DistributedGraph::partition_edges_by_rank(&info, &edges);

        // assert that
        assert_eq!(
            buf,
            vec![
                Edge::from((0, 1, 1.0)),
                Edge::from((1, 0, 1.0)),
                Edge::from((2, 3, 2.0)),
                Edge::from((3, 2, 2.0))
            ]
        );
        assert_eq!(counts, vec![4, 0]);
        assert_eq!(displs, vec![0, 4]);
    }

    #[test]
    fn test_partition_edges_different_ranks() {
        // Assumption: Edge::from(s) belonging to different ranks should be correctly distributed
        let info = MockInfoBuilder::new()
            .size(3)
            .global_vcount(6)
            .build()
            .unwrap();
        let edges = vec![
            Edge::from((0, 3, 1.0)),
            Edge::from((1, 4, 2.0)),
            Edge::from((2, 5, 3.0)),
        ];

        let (buf, counts, displs) = DistributedGraph::partition_edges_by_rank(&info, &edges);

        assert_eq!(
            buf,
            vec![
                Edge::from((0, 3, 1.0)),
                Edge::from((1, 4, 2.0)),
                Edge::from((2, 5, 3.0)),
                Edge::from((3, 0, 1.0)),
                Edge::from((4, 1, 2.0)),
                Edge::from((5, 2, 3.0))
            ]
        );
        assert_eq!(counts, vec![2, 2, 2]);
        assert_eq!(displs, vec![0, 2, 4]);
    }

    #[test]
    fn test_partition_edges_with_self_loops() {
        // Assumption: Self-loops should only be added once to the buffer
        let info = MockInfoBuilder::new()
            .size(2)
            .global_vcount(2)
            .build()
            .unwrap();
        let edges = vec![Edge::from((0, 0, 1.0)), Edge::from((1, 1, 2.0))];

        let (buf, counts, displs) = DistributedGraph::partition_edges_by_rank(&info, &edges);

        assert_eq!(buf, vec![Edge::from((0, 0, 1.0)), Edge::from((1, 1, 2.0))]);
        assert_eq!(counts, vec![1, 1]);
        assert_eq!(displs, vec![0, 1]);
    }

    #[test]
    fn test_partition_edges_sorted_output() {
        // Assumption: The output buffer should be sorted first by rank, then by source vertex
        let info = MockInfoBuilder::new()
            .size(2)
            .global_vcount(3)
            .build()
            .unwrap();
        let edges = vec![
            Edge::from((2, 0, 1.0)),
            Edge::from((0, 1, 2.0)),
            Edge::from((1, 2, 3.0)),
        ];

        let (buf, counts, displs) = DistributedGraph::partition_edges_by_rank(&info, &edges);

        assert_eq!(
            buf,
            vec![
                Edge::from((0, 1, 2.0)),
                Edge::from((0, 2, 1.0)),
                Edge::from((1, 0, 2.0)),
                Edge::from((1, 2, 3.0)),
                Edge::from((2, 0, 1.0)),
                Edge::from((2, 1, 3.0))
            ]
        );
        assert_eq!(counts, vec![4, 2]);
        assert_eq!(displs, vec![0, 4]);
    }

    #[test]
    fn test_partition_edges_empty_input() {
        // Assumption: The function should handle empty input correctly
        let info = MockInfoBuilder::new()
            .size(2)
            .global_vcount(4)
            .build()
            .unwrap();
        let edges = vec![];

        let (buf, counts, displs) = DistributedGraph::partition_edges_by_rank(&info, &edges);

        assert_eq!(buf, vec![]);
        assert_eq!(counts, vec![0, 0]);
        assert_eq!(displs, vec![0, 0]);
    }
}
