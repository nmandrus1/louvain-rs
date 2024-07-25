use std::collections::HashSet;

use indexmap::IndexMap;

use crate::MessageRouter;

use super::{LouvainGraph, MessageManager, Owned, OwnershipInfo};

use mpi::traits::Equivalence;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Equivalence)]
pub struct CommunityID(pub usize);

impl From<usize> for CommunityID {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl Owned for CommunityID {
    fn owner<I: OwnershipInfo>(&self, info: &I) -> mpi::Rank {
        info.owner_of_community(*self)
    }
}

/// Represents a Community in our Graph
pub struct Community {
    id: CommunityID,
    internal_weight: f64,
    total_weight: f64,
    vertices: HashSet<usize>,
}

impl Community {
    /// initialize a community from a vertex
    fn from_vtx(id: CommunityID, total_weight: f64) -> Self {
        Self {
            id,
            internal_weight: 0.0,
            total_weight,
            vertices: HashSet::new(),
        }
    }
}

/// struct responsible for carrying out distributed Louvain Method
pub struct DistributedCommunityDetection<'a, G: LouvainGraph, R: MessageRouter> {
    graph: &'a G,
    owned_communities: IndexMap<usize, Community>,
    message_manager: MessageManager<'a, R>,
}

impl<'a, G: LouvainGraph, R: MessageRouter> DistributedCommunityDetection<'a, G, R> {
    pub fn new(graph: &'a G, router: &'a R) -> Self {
        let mut owned_communities = IndexMap::with_capacity(graph.local_vertex_count());
        graph.vertices().for_each(|v| {
            owned_communities
                .entry(v)
                .or_insert(Community::from_vtx(v.into(), graph.weighted_degree(v)));
        });

        Self {
            graph,
            owned_communities,
            message_manager: MessageManager::new(router),
        }
    }

    pub fn one_level(&mut self) {}

    /// calculate modlarity as the sum of each community's contribution
    fn local_modularity(&self) -> f64 {
        let m2 = self.graph.global_edge_count() as f64 * 2.0;

        self.owned_communities
            .iter()
            .filter_map(|(_, comm)| {
                (comm.total_weight > 0.0)
                    .then_some((comm.internal_weight / m2) - (comm.total_weight / m2).powi(2))
            })
            .sum()
    }

    pub fn global_modularity(&self) -> f64 {
        let partial = self.local_modularity();
        self.message_manager.global_modularity(partial)
    }
}
