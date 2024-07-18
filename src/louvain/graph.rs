use std::collections::HashMap;

use petgraph::csr::Csr;

use anyhow::anyhow;

use mpi::{
    collective::SystemOperation,
    datatype::Partition,
    environment::Universe,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence, PartitionedBuffer},
    Count, Rank,
};

#[derive(Equivalence, Clone, Copy)]
struct Edge(usize, usize, f32);

/// Stores information about our distributed setup
struct DistributedInfo {
    world: SimpleCommunicator,
    rank: usize,
    /// size of MPI_COMM_WORLD
    size: usize,
    /// rows of the global adj matrix
    owned_rows: (usize, usize),
    /// idx = rank & element rows for that rank exclusive rows.0..rows.1
    row_ownership: Vec<(usize, usize)>,
}

impl DistributedInfo {
    /// Given the size of the MPI topology, and the global number of vertices, compute the row
    /// ownership of every rank
    fn compute_vertex_distribution(size: usize, global_vcount: usize) -> Vec<(usize, usize)> {
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

    /// Return the number of vertices this process owns
    fn local_vcount(&self) -> usize {
        self.owned_rows.1 - self.owned_rows.0
    }

    fn init(universe: &Universe, global_vcount: usize) -> Self {
        let world = universe.world();
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
        }
    }
}

/// Simple wrapper over petgraph CSR to represent a Distributed Graph
pub struct DistributedGraph<N, E> {
    inner: Csr<N, E>,

    global_vcount: usize,
    local_vcount: usize,

    universe: Universe,
    info: DistributedInfo,
}

impl<N, E> DistributedGraph<N, E> {
    /// given a vertex with a global id, return the rank owner of that vertex
    pub fn rank_owner(&self, vtx: usize) -> Rank {
        (vtx / self.local_vcount) as i32
    }

    /// Builds the graph by communicating with MPI Processes
    pub fn from_distributed(edges: &[Edge]) -> anyhow::Result<Self> {
        let universe = mpi::initialize().ok_or(anyhow::anyhow!("MPI Not Initialized"))?;
        let world = universe.world();
        let rank = world.rank();
        let comm_size = world.size() as usize;

        let global_vcount = Self::compute_global_vcount(&edges, &world)?;

        let info = DistributedInfo::init(&universe, global_vcount);
        let local_vcount = info.local_vcount();
    }

    // Helper functions

    /// given the edge list, communicate with all ranks and determine the global vcount
    fn compute_global_vcount(edges: &[Edge], world: &SimpleCommunicator) -> anyhow::Result<usize> {
        let local_max_vtx = edges
            .iter()
            .map(|e| std::cmp::max(e.0, e.1))
            .max()
            .ok_or(anyhow::anyhow!("Unable to find maximum vertex"))?;

        // communicate max vtx and determine how many vertices are in the total distributed graph
        let mut global_max_vtx: usize = usize::MAX;
        world.all_reduce_into(&local_max_vtx, &mut global_max_vtx, SystemOperation::max());

        // error checking: return global_max_vtx + 1 if MPI call was successful
        // or an error if it was not set properly
        (global_max_vtx != usize::MAX)
            .then_some(global_max_vtx + 1)
            .ok_or(anyhow!("Failed to determine global vcount"))
    }

    /// Takes a list of edges, and sorts them into a 2d vector of messages to send to each rank
    fn organize_edges_into_message(
        info: &DistributedInfo,
        edges: &[Edge],
    ) -> impl PartitionedBuffer {
        // organize edges by rank
        let mut counts = vec![0; info.size];
        let mut msg_map = Vec::<Vec<Edge>>::with_capacity(info.size);

        edges.iter().for_each(|edge| {
            // TODO: use info to determine rank owner safely and robustly
            let rank1 = edge.0 / info.local_vcount();
            let rank2 = edge.1 / info.local_vcount();

            msg_map[rank1].push(*edge);
            counts[rank1] += 1;

            msg_map[rank2].push(Edge(edge.1, edge.0, edge.2));
            counts[rank2] += 1;
        });

        let displs: Vec<Count> = counts
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();

        let buf: Vec<Edge> = msg_map.into_iter().flatten().collect();
        let partition = Partition::new(&buf, counts, displs);
        todo!("Finish this!!")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
