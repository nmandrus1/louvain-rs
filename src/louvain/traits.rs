use indexmap::IndexSet;
use petgraph::{
    visit::{EdgeRef, IntoEdgeReferences, IntoEdges, IntoNeighbors, IntoNodeReferences},
    Directed,
};

use super::{Community, CommunityID, IterationStep, LouvainMessage, VertexID};
use mpi::{topology::Process, Count, Rank};

/// Router must be able to determine ownership of Communities and Vertices
pub trait MessageRouter {
    fn route<S: CommunityState>(&self, message: LouvainMessage, state: &S) -> IndexSet<Rank>;
    fn process_at_rank(&self, rank: Rank) -> Process;
    fn this_process(&self) -> Process;

    // number of processes available to send messages to
    fn size(&self) -> usize;

    // send
    fn distribute_counts(&self, buf: &[Count], recv_counts: &mut [Count]) -> anyhow::Result<()>;

    fn global_modularity(&self, partial: f64) -> f64;
}

/// Trait with functions needed to able to run Community Detection algorithm
pub trait LouvainGraph {
    fn local_vertex_count(&self) -> usize;
    fn global_vertex_count(&self) -> usize;
    fn global_edge_count(&self) -> usize;
    fn weighted_degree(&self, vertex: usize) -> f64;

    // Iterators
    fn neighbors(&self, vertex: VertexID) -> impl Iterator<Item = (VertexID, f64)>; // (neighbor, edge_weight)
    /// iterator over the vertices owned by this graph
    fn vertices(&self) -> impl Iterator<Item = VertexID>;

    /// feed vertices to the driver of the louvain algorithm
    fn vertex_feed(&self) -> impl Iterator<Item = IterationStep>;

    // determine ownwership
    fn owner_of_vertex(&self, vtx: VertexID) -> Rank;
    fn owner_of_community(&self, community: CommunityID) -> Rank;
    fn is_local_vertex(&self, vertex: usize) -> bool;
}

use super::{CommunityInfo, NeighborInfo, VertexMovement};

pub trait MessageManagerTrait {
    fn queue_community_update<S: CommunityState>(&mut self, update: CommunityInfo, state: &S);
    fn queue_vertex_movement<S: CommunityState>(&mut self, movement: VertexMovement, state: &S);
    fn queue_neighbor_info<S: CommunityState>(&mut self, info: NeighborInfo, state: &S);

    fn exchange_vertex_movements(&mut self) -> anyhow::Result<()>;
    fn exchange_other_messages(&mut self) -> anyhow::Result<()>;

    fn get_received_community_updates(&self) -> impl Iterator<Item = &CommunityInfo>;
    fn get_received_vertex_movements(&self) -> impl Iterator<Item = &VertexMovement>;
    fn get_received_neighbor_info(&self) -> impl Iterator<Item = &NeighborInfo>;

    fn clear(&mut self);
}

pub trait CommunityState {
    fn get_community_members(&self, id: &CommunityID) -> impl Iterator<Item = &VertexID>;

    fn local_communities(&self) -> impl Iterator<Item = &CommunityID>;

    fn neighboring_communities<G: LouvainGraph>(
        &self,
        id: VertexID,
        graph: &G,
    ) -> impl Iterator<Item = (CommunityID, f64)>;

    fn update_community(&mut self, update: &CommunityInfo) -> bool;

    fn move_vertex(&mut self, movement: &VertexMovement);

    fn get_updated_communities(&self) -> impl Iterator<Item = &CommunityID>;

    fn get_local_community(&self, id: &CommunityID) -> &Community;

    fn community_of(&self, vtx: &VertexID) -> &CommunityID;

    // batch updating
    fn batch_process_vertex_movements<'recv>(
        &mut self,
        movements: impl Iterator<Item = &'recv VertexMovement>,
    );
    fn batch_process_community_update<'recv>(
        &mut self,
        updates: impl Iterator<Item = &'recv CommunityInfo>,
    );
    fn batch_process_neighor_info<'recv>(
        &mut self,
        info: impl Iterator<Item = &'recv NeighborInfo>,
    );
}

#[cfg(test)]
impl LouvainGraph for petgraph::csr::Csr<VertexID, f64, Directed, usize> {
    fn local_vertex_count(&self) -> usize {
        self.node_count()
    }

    fn global_vertex_count(&self) -> usize {
        self.node_count()
    }

    fn global_edge_count(&self) -> usize {
        self.edge_count()
    }

    fn weighted_degree(&self, vertex: usize) -> f64 {
        self.edges(vertex).map(|e| e.weight()).sum::<f64>()
    }

    fn neighbors(&self, vertex: VertexID) -> impl Iterator<Item = (VertexID, f64)> {
        self.edges(vertex.0)
            .map(|e| (e.target().into(), *e.weight()))
    }

    fn vertices(&self) -> impl Iterator<Item = VertexID> {
        self.node_references().map(|n| n.0.into())
    }

    fn vertex_feed(&self) -> impl Iterator<Item = IterationStep> {
        self.vertices().map(|v| IterationStep::Process(v))
    }

    fn owner_of_vertex(&self, _: VertexID) -> Rank {
        0
    }

    fn owner_of_community(&self, _: CommunityID) -> Rank {
        0
    }

    fn is_local_vertex(&self, _: usize) -> bool {
        true
    }
}
