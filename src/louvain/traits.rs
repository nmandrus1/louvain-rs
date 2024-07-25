use std::ops::Add;

use super::{CommunityID, LouvainMessage, VertexID};
use mpi::{topology::Process, Count, Rank};

/// Used to determine the Process that owns a vertex or community.
/// the functions are split up in case strategies change for determining either one
pub trait OwnershipInfo {
    fn owner_of_vertex(&self, vtx: VertexID) -> Rank;
    fn owner_of_community(&self, community: CommunityID) -> Rank;
}

/// indicates that a particular value is associated with a owner process
pub trait Owned {
    fn owner<I: OwnershipInfo>(&self, info: &I) -> Rank;
}

/// Router must be able to determine ownership of Communities and Vertices
pub trait MessageRouter {
    fn route(&self, message: LouvainMessage) -> Vec<Rank>;
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
    fn neighbors(&self, vertex: usize) -> impl Iterator<Item = (usize, f64)>; // (neighbor, edge_weight)
    /// iterator over the vertices owned by this graph
    fn vertices(&self) -> impl Iterator<Item = usize>;
    fn is_local_vertex(&self, vertex: usize) -> bool;
}

use super::{CommunityUpdate, NeighborInfo, VertexMovement};

pub trait MessageManagerTrait {
    fn queue_community_update(&mut self, update: CommunityUpdate);
    fn queue_vertex_movement(&mut self, movement: VertexMovement);
    fn queue_neighbor_info(&mut self, info: NeighborInfo);

    fn exchange_vertex_movements(&mut self) -> anyhow::Result<()>;
    fn exchange_other_messages(&mut self) -> anyhow::Result<()>;

    fn get_received_community_updates(&self) -> impl Iterator<Item = &CommunityUpdate>;
    fn get_received_vertex_movements(&self) -> impl Iterator<Item = &VertexMovement>;
    fn get_received_neighbor_info(&self) -> impl Iterator<Item = &NeighborInfo>;

    fn clear(&mut self);
}
