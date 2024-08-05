use indexmap::{indexset, IndexMap, IndexSet};

use super::{
    CommunityInfo, CommunityOperation, CommunityState, LouvainGraph, NeighborInfo, VertexID,
    VertexMovement,
};

use mpi::traits::Equivalence;

use log::{debug, error, info};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Equivalence, Hash)]
pub struct CommunityID(pub usize);

impl From<usize> for CommunityID {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

/// Represents a Community in our Graph
#[derive(Debug, Default)]
pub struct Community {
    id: CommunityID,
    pub internal_weight: f64,
    pub total_weight: f64,
    vertices: IndexSet<VertexID>,
    is_local: bool,
}

impl Community {
    /// initialize a community from a vertex
    fn from_vtx(id: VertexID, total_weight: f64) -> Self {
        Self {
            id: CommunityID(id.0),
            internal_weight: 0.0,
            total_weight,
            vertices: indexset! {id},
            is_local: true,
        }
    }

    pub fn get_info(&self) -> CommunityInfo {
        CommunityInfo {
            id: self.id,
            internal_weight: self.internal_weight,
            total_weight: self.total_weight,
        }
    }

    fn from_info(info: &CommunityInfo) -> Self {
        Self {
            id: info.id,
            internal_weight: info.internal_weight,
            total_weight: info.total_weight,
            vertices: IndexSet::new(),
            is_local: false,
        }
    }
}

/// Manages the State of Community Detection
/// Intentionally separated from the communication logic
pub struct StateManager {
    /// tracks local AND remote communities
    tracked_communities: IndexMap<CommunityID, Community>,
    /// tracks our local vertices and which communities they belong to
    vtx_community_map: IndexMap<VertexID, CommunityID>,
    updated_communities: IndexSet<CommunityID>,
}

impl StateManager {
    pub fn new<G: LouvainGraph>(graph: &G) -> Self {
        let mut tracked_communities = IndexMap::with_capacity(graph.local_vertex_count());
        let mut vtx_community_map = IndexMap::with_capacity(graph.local_vertex_count());
        let updated_communities = IndexSet::with_capacity(graph.local_vertex_count());

        graph.vertices().for_each(|v| {
            let comm = Community::from_vtx(v, graph.weighted_degree(v.0));
            let id = comm.id;
            tracked_communities.insert(id, comm);
            vtx_community_map.insert(v, id);

            debug!("Initialized community for vertex {:?}", v);
        });

        Self {
            tracked_communities,
            vtx_community_map,
            updated_communities,
        }
    }

    // Helper methods

    fn insert(
        comm: &mut Community,
        vtx_map: &mut IndexMap<VertexID, CommunityID>,
        movement: &VertexMovement,
    ) {
        comm.internal_weight += 2.0 * movement.internal_weight;
        comm.total_weight += movement.total_weight;
        vtx_map.insert(movement.vtx, comm.id);

        // if the vertex was already in the set then signal a warning, we shouldn't
        // be inserting a vertex twice
        if !comm.vertices.insert(movement.vtx) {
            panic!(
                "Vertex {:?} inserted twice into Community: {:?}",
                movement.vtx, movement.community_id
            )
        }
    }

    fn remove(
        comm: &mut Community,
        vtx_map: &mut IndexMap<VertexID, CommunityID>,
        movement: &VertexMovement,
    ) {
        comm.internal_weight -= 2.0 * movement.internal_weight;
        comm.total_weight -= movement.total_weight;
        vtx_map.swap_remove(&movement.vtx);

        // if the vertex was already in the set then signal a warning, we shouldn't
        // be inserting a vertex twice
        if !comm.vertices.swap_remove(&movement.vtx) {
            panic!(
                "Vertex {:?} removed from community {:?} but was not in the hashset",
                movement.vtx, movement.community_id
            )
        }
    }
}

impl CommunityState for StateManager {
    /// Iterator over the memebers of a community, panics if the community is not in the map
    fn get_community_members(&self, id: &CommunityID) -> impl Iterator<Item = &VertexID> {
        if let Some(comm) = self.tracked_communities.get(id) {
            debug_assert!(!comm.vertices.is_empty());
            comm.vertices.iter()
        } else {
            error!("Community id {:?} not located in map", id);
            panic!("Community id {:?} not located in map", id);
        }
    }

    /// returns a iterator with each element containing a neighboring community and the sum of edge weights to that community
    fn neighboring_communities<G: LouvainGraph>(
        &self,
        id: VertexID,
        graph: &G,
    ) -> impl Iterator<Item = (CommunityID, f64)> {
        use std::ops::AddAssign;

        let mut map = IndexMap::new();
        graph
            .neighbors(id)
            .map(|(neighbor, weight)| (self.vtx_community_map[&neighbor], weight))
            // .chain(once((self.vtx_community_map[&id], 0.0)))
            .for_each(|(id, weight)| map.entry(id).or_insert(0.0).add_assign(weight));

        map.into_iter()
    }

    /// overwrite community information with passed upate or create a new Community
    /// if it didn't exist in our table. Returns true if community was overwritten, false
    /// if the community didn't exist. The only communities we should be recieving updates on should
    /// be remote. This function panics if the community being updated is local
    fn update_community(&mut self, update: &CommunityInfo) -> bool {
        match self.tracked_communities.get_mut(&update.id) {
            Some(comm) => {
                // debug_assert!(!comm.is_local);

                comm.total_weight = update.total_weight;
                comm.internal_weight = update.internal_weight;
                info!("Community {:?} updated", comm.id);
                true
            }
            None => {
                self.tracked_communities
                    .insert(update.id, Community::from_info(update));
                info!(
                    "Community {:?} added to map with in: {}, total: {}",
                    update.id, update.internal_weight, update.total_weight
                );
                false
            }
        }
    }

    /// moves a vertex into a community
    fn move_vertex(&mut self, movement: &VertexMovement) {
        debug!("{:?}", movement);
        if let Some(comm) = self.tracked_communities.get_mut(&movement.community_id) {
            let vtx_map = &mut self.vtx_community_map;

            match movement.operation {
                CommunityOperation::Insert => Self::insert(comm, vtx_map, &movement),
                CommunityOperation::Remove => Self::remove(comm, vtx_map, &movement),
            }

            self.updated_communities.insert(comm.id);
        } else {
            error!(
                "Attempted to move vertex {:?} into community that doesn't exist: {:?}",
                movement.vtx, movement.community_id
            );
            panic!(
                "Attempted to move vertex {:?} into community that doesn't exist: {:?}",
                movement.vtx, movement.community_id
            )
        }
    }

    /// returns an iterator over all the communities that have been updated
    fn get_updated_communities(&self) -> impl Iterator<Item = &CommunityID> {
        self.updated_communities.iter()
        // TODO: how are we going to handle clearing this?
    }

    #[inline]
    fn get_local_community(&self, id: &CommunityID) -> &Community {
        if let Some(comm) = self.tracked_communities.get(id) {
            debug_assert!(comm.is_local);

            comm
        } else {
            panic!("Community {:?} not being tracked", id);
        }
    }

    fn local_communities(&self) -> impl Iterator<Item = &CommunityID> {
        // TODO make this better
        self.tracked_communities
            .iter()
            .filter(|(_, comm)| comm.is_local)
            .map(|(id, _)| id)
    }

    fn community_of(&self, vtx: &VertexID) -> &CommunityID {
        if let Some(id) = self.vtx_community_map.get(vtx) {
            id
        } else {
            panic!("Vertex {:?} not in the vertex to community map", vtx);
        }
    }

    #[inline]
    fn batch_process_vertex_movements<'recv>(
        &mut self,
        movements: impl Iterator<Item = &'recv VertexMovement>,
    ) {
        movements.for_each(|m| self.move_vertex(m))
    }

    #[inline]
    fn batch_process_community_update<'recv>(
        &mut self,
        updates: impl Iterator<Item = &'recv CommunityInfo>,
    ) {
        updates.for_each(|u| {
            self.update_community(u);
        })
    }

    #[inline]
    fn batch_process_neighor_info<'recv>(
        &mut self,
        info: impl Iterator<Item = &'recv NeighborInfo>,
    ) {
        info.for_each(|update| {
            let comm = self.vtx_community_map.get_mut(&update.vertex_id).unwrap();
            *comm = update.community_id;
        })
    }
}

// #[cfg(test)]
// mod tests {

//     use super::*;

//     use petgraph::{csr::Csr, Directed};

//     // Mock LouvainGraph implementation for testing
//     type MockGraph = Csr<VertexID, f64, Directed, usize>;

//     #[test]
//     fn test_state_manager_initialization() {
//         let graph = MockGraph;
//         let state_manager = StateManager::new(&graph);

//         assert_eq!(state_manager.tracked_communities.len(), 5);
//         assert_eq!(state_manager.vtx_community_map.len(), 5);
//         assert_eq!(state_manager.updated_communities.len(), 0);

//         for i in 0..5 {
//             let vtx = VertexID(i);
//             let comm_id = CommunityID(i);
//             assert!(state_manager.tracked_communities.contains_key(&comm_id));
//             assert_eq!(state_manager.vtx_community_map.get(&vtx), Some(&comm_id));

//             let comm = state_manager.tracked_communities.get(&comm_id).unwrap();
//             assert_eq!(comm.id, comm_id);
//             assert_eq!(comm.internal_weight, 0.0);
//             assert_eq!(comm.total_weight, 2.0);
//             assert!(comm.vertices.contains(&vtx));
//             assert!(comm.is_local);
//         }
//     }

//     #[test]
//     fn test_get_community_members() {
//         let graph = MockGraph;
//         let state_manager = StateManager::new(&graph);

//         let members: Vec<_> = state_manager
//             .get_community_members(&CommunityID(0))
//             .collect();
//         assert_eq!(members, vec![&VertexID(0)]);

//         // Test for non-existent community
//         let result = std::panic::catch_unwind(|| {
//             state_manager.get_community_members(&CommunityID(10));
//         });
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_neighboring_communities() {
//         let graph = MockGraph;
//         let state_manager = StateManager::new(&graph);

//         let neighbors: Vec<_> = state_manager
//             .neighboring_communities(VertexID(0), &graph)
//             .collect();
//         assert_eq!(
//             neighbors,
//             vec![(CommunityID(1), 1.0), (CommunityID(2), 1.0)]
//         );
//     }

//     #[test]
//     fn test_update_community() {
//         let graph = MockGraph;
//         let mut state_manager = StateManager::new(&graph);

//         // Update existing community
//         let update = CommunityInfo {
//             id: CommunityID(0),
//             internal_weight: 1.5,
//             total_weight: 3.0,
//         };
//         assert!(state_manager.update_community(&update));

//         let updated_comm = state_manager
//             .tracked_communities
//             .get(&CommunityID(0))
//             .unwrap();
//         assert_eq!(updated_comm.internal_weight, 1.5);
//         assert_eq!(updated_comm.total_weight, 3.0);

//         // Add new community
//         let new_update = CommunityInfo {
//             id: CommunityID(10),
//             internal_weight: 2.0,
//             total_weight: 4.0,
//         };
//         assert!(!state_manager.update_community(&new_update));

//         let new_comm = state_manager
//             .tracked_communities
//             .get(&CommunityID(10))
//             .unwrap();
//         assert_eq!(new_comm.internal_weight, 2.0);
//         assert_eq!(new_comm.total_weight, 4.0);
//         assert!(!new_comm.is_local);
//     }

//     #[test]
//     fn test_move_vertex() {
//         let graph = MockGraph;
//         let mut state_manager = StateManager::new(&graph);

//         // Move vertex 0 to community 1
//         let movement = VertexMovement {
//             community_id: CommunityID(1),
//             vtx: VertexID(0),
//             internal_weight: 1.0,
//             total_weight: 2.0,
//             operation: CommunityOperation::Insert,
//         };
//         state_manager.move_vertex(&movement);

//         assert_eq!(
//             state_manager.vtx_community_map.get(&VertexID(0)),
//             Some(&CommunityID(1))
//         );

//         let comm1 = state_manager
//             .tracked_communities
//             .get(&CommunityID(1))
//             .unwrap();
//         assert_eq!(comm1.internal_weight, 2.0);
//         assert_eq!(comm1.total_weight, 4.0);
//         assert!(comm1.vertices.contains(&VertexID(0)));

//         assert!(state_manager.updated_communities.contains(&CommunityID(1)));

//         // Remove vertex 0 from community 1
//         let remove_movement = VertexMovement {
//             community_id: CommunityID(1),
//             vtx: VertexID(0),
//             internal_weight: 1.0,
//             total_weight: 2.0,
//             operation: CommunityOperation::Remove,
//         };
//         state_manager.move_vertex(&remove_movement);

//         let comm1_after_remove = state_manager
//             .tracked_communities
//             .get(&CommunityID(1))
//             .unwrap();
//         assert_eq!(comm1_after_remove.internal_weight, 0.0);
//         assert_eq!(comm1_after_remove.total_weight, 2.0);
//         assert!(!comm1_after_remove.vertices.contains(&VertexID(0)));
//     }

//     #[test]
//     fn test_get_updated_communities() {
//         let graph = MockGraph;
//         let mut state_manager = StateManager::new(&graph);

//         // Update a community
//         let movement = VertexMovement {
//             community_id: CommunityID(1),
//             vtx: VertexID(0),
//             internal_weight: 1.0,
//             total_weight: 2.0,
//             operation: CommunityOperation::Insert,
//         };
//         state_manager.move_vertex(&movement);

//         let updated: Vec<_> = state_manager.get_updated_communities().collect();
//         assert_eq!(updated, vec![&CommunityID(1)]);
//     }

//     #[test]
//     fn test_get_local_community() {
//         let graph = MockGraph;
//         let state_manager = StateManager::new(&graph);

//         let local_comm = state_manager.get_local_community(&CommunityID(0));
//         assert_eq!(local_comm.id, CommunityID(0));
//         assert!(local_comm.is_local);

//         // Test for non-existent community
//         let result = std::panic::catch_unwind(|| {
//             state_manager.get_local_community(&CommunityID(10));
//         });
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_local_communities() {
//         let graph = MockGraph;
//         let state_manager = StateManager::new(&graph);

//         let local_comms: Vec<_> = state_manager.local_communities().collect();
//         assert_eq!(
//             local_comms,
//             vec![
//                 &CommunityID(0),
//                 &CommunityID(1),
//                 &CommunityID(2),
//                 &CommunityID(3),
//                 &CommunityID(4)
//             ]
//         );
//     }

//     #[test]
//     fn test_community_of() {
//         let graph = MockGraph;
//         let state_manager = StateManager::new(&graph);

//         assert_eq!(state_manager.community_of(&VertexID(0)), &CommunityID(0));
//         assert_eq!(state_manager.community_of(&VertexID(4)), &CommunityID(4));

//         // Test for non-existent vertex
//         let result = std::panic::catch_unwind(|| {
//             state_manager.community_of(&VertexID(10));
//         });
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_batch_process_vertex_movements() {
//         let graph = MockGraph;
//         let mut state_manager = StateManager::new(&graph);

//         let movements = vec![
//             VertexMovement {
//                 community_id: CommunityID(1),
//                 vtx: VertexID(0),
//                 internal_weight: 1.0,
//                 total_weight: 2.0,
//                 operation: CommunityOperation::Insert,
//             },
//             VertexMovement {
//                 community_id: CommunityID(2),
//                 vtx: VertexID(1),
//                 internal_weight: 1.5,
//                 total_weight: 3.0,
//                 operation: CommunityOperation::Insert,
//             },
//         ];

//         state_manager.batch_process_vertex_movements(movements.iter());

//         assert_eq!(
//             state_manager.vtx_community_map.get(&VertexID(0)),
//             Some(&CommunityID(1))
//         );
//         assert_eq!(
//             state_manager.vtx_community_map.get(&VertexID(1)),
//             Some(&CommunityID(2))
//         );

//         let comm1 = state_manager
//             .tracked_communities
//             .get(&CommunityID(1))
//             .unwrap();
//         assert_eq!(comm1.internal_weight, 2.0);
//         assert_eq!(comm1.total_weight, 4.0);

//         let comm2 = state_manager
//             .tracked_communities
//             .get(&CommunityID(2))
//             .unwrap();
//         assert_eq!(comm2.internal_weight, 3.0);
//         assert_eq!(comm2.total_weight, 5.0);

//         assert!(state_manager.updated_communities.contains(&CommunityID(1)));
//         assert!(state_manager.updated_communities.contains(&CommunityID(2)));
//     }

//     #[test]
//     fn test_batch_process_community_update() {
//         let graph = MockGraph;
//         let mut state_manager = StateManager::new(&graph);

//         let updates = vec![
//             CommunityInfo {
//                 id: CommunityID(0),
//                 internal_weight: 1.5,
//                 total_weight: 3.0,
//             },
//             CommunityInfo {
//                 id: CommunityID(10),
//                 internal_weight: 2.0,
//                 total_weight: 4.0,
//             },
//         ];

//         state_manager.batch_process_community_update(updates.iter());

//         let updated_comm0 = state_manager
//             .tracked_communities
//             .get(&CommunityID(0))
//             .unwrap();
//         assert_eq!(updated_comm0.internal_weight, 1.5);
//         assert_eq!(updated_comm0.total_weight, 3.0);

//         let new_comm10 = state_manager
//             .tracked_communities
//             .get(&CommunityID(10))
//             .unwrap();
//         assert_eq!(new_comm10.internal_weight, 2.0);
//         assert_eq!(new_comm10.total_weight, 4.0);
//         assert!(!new_comm10.is_local);
//     }

//     #[test]
//     fn test_batch_process_neighbor_info() {
//         let graph = MockGraph;
//         let mut state_manager = StateManager::new(&graph);

//         let neighbor_info = vec![
//             NeighborInfo {
//                 vertex_id: VertexID(0),
//                 community_id: CommunityID(1),
//             },
//             NeighborInfo {
//                 vertex_id: VertexID(1),
//                 community_id: CommunityID(2),
//             },
//         ];

//         state_manager.batch_process_neighor_info(neighbor_info.iter());

//         assert_eq!(
//             state_manager.vtx_community_map.get(&VertexID(0)),
//             Some(&CommunityID(1))
//         );
//         assert_eq!(
//             state_manager.vtx_community_map.get(&VertexID(1)),
//             Some(&CommunityID(2))
//         );
//     }
// }

#[cfg(test)]
mod tests {
    use crate::logger;

    use super::*;
    use petgraph::{csr::Csr, Directed};

    type MockGraph = Csr<VertexID, f64, Directed, usize>;

    fn create_test_graph() -> MockGraph {
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 0, 1.0),
            (1, 2, 3.0),
            (1, 3, 4.0),
            (2, 0, 2.0),
            (2, 1, 3.0),
            (2, 3, 5.0),
            (3, 1, 4.0),
            (3, 2, 5.0),
            (3, 4, 6.0),
            (4, 3, 6.0),
        ];
        Csr::from_sorted_edges(&edges).unwrap()
    }

    fn create_large_test_graph() -> MockGraph {
        let mut edges = Vec::new();
        for i in 0..100 {
            for j in i + 1..100 {
                edges.push((i, j, (i + j) as f64 / 100.0));
            }
        }
        Csr::from_sorted_edges(&edges).unwrap()
    }

    #[test]
    fn test_state_manager_initialization() {
        let graph = create_test_graph();
        let state_manager = StateManager::new(&graph);

        assert_eq!(state_manager.tracked_communities.len(), 5);
        assert_eq!(state_manager.vtx_community_map.len(), 5);
        assert_eq!(state_manager.updated_communities.len(), 0);

        for i in 0..5 {
            let vtx = VertexID(i);
            let comm_id = CommunityID(i);
            assert!(state_manager.tracked_communities.contains_key(&comm_id));
            assert_eq!(state_manager.vtx_community_map.get(&vtx), Some(&comm_id));

            let comm = state_manager.tracked_communities.get(&comm_id).unwrap();
            assert_eq!(comm.id, comm_id);
            assert_eq!(comm.internal_weight, 0.0);
            assert_eq!(comm.total_weight, graph.weighted_degree(i));
            assert!(comm.vertices.contains(&vtx));
            assert!(comm.is_local);
        }
    }

    #[test]
    fn test_get_community_members() {
        let graph = create_test_graph();
        let state_manager = StateManager::new(&graph);

        for i in 0..5 {
            let members: Vec<_> = state_manager
                .get_community_members(&CommunityID(i))
                .collect();
            assert_eq!(members, vec![&VertexID(i)]);
        }

        // Test for non-existent community
        let result = std::panic::catch_unwind(|| {
            state_manager.get_community_members(&CommunityID(10));
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_neighboring_communities() {
        let graph = create_test_graph();
        let state_manager = StateManager::new(&graph);

        let neighbors: Vec<_> = state_manager
            .neighboring_communities(VertexID(0), &graph)
            .collect();
        assert_eq!(
            neighbors,
            vec![(CommunityID(1), 1.0), (CommunityID(2), 2.0)]
        );

        let neighbors: Vec<_> = state_manager
            .neighboring_communities(VertexID(3), &graph)
            .collect();
        assert_eq!(
            neighbors,
            vec![
                (CommunityID(1), 4.0),
                (CommunityID(2), 5.0),
                (CommunityID(4), 6.0)
            ]
        );
    }

    #[test]
    fn test_update_community() {
        let graph = create_test_graph();
        let mut state_manager = StateManager::new(&graph);

        // Update existing community
        let update = CommunityInfo {
            id: CommunityID(0),
            internal_weight: 1.5,
            total_weight: 3.0,
        };
        assert!(state_manager.update_community(&update));

        let updated_comm = state_manager
            .tracked_communities
            .get(&CommunityID(0))
            .unwrap();
        assert_eq!(updated_comm.internal_weight, 1.5);
        assert_eq!(updated_comm.total_weight, 3.0);

        // Add new community
        let new_update = CommunityInfo {
            id: CommunityID(10),
            internal_weight: 2.0,
            total_weight: 4.0,
        };
        assert!(!state_manager.update_community(&new_update));

        let new_comm = state_manager
            .tracked_communities
            .get(&CommunityID(10))
            .unwrap();
        assert_eq!(new_comm.internal_weight, 2.0);
        assert_eq!(new_comm.total_weight, 4.0);
        assert!(!new_comm.is_local);
    }

    #[test]
    fn test_move_vertex() {
        let graph = create_test_graph();
        let mut state_manager = StateManager::new(&graph);

        // Move vertex 0 to community 1
        let movement = VertexMovement {
            community_id: CommunityID(1),
            vtx: VertexID(0),
            internal_weight: 1.0,
            total_weight: 3.0,
            operation: CommunityOperation::Insert,
        };
        state_manager.move_vertex(&movement);

        assert_eq!(
            state_manager.vtx_community_map.get(&VertexID(0)),
            Some(&CommunityID(1))
        );

        let comm1 = state_manager
            .tracked_communities
            .get(&CommunityID(1))
            .unwrap();
        assert_eq!(comm1.internal_weight, 2.0);
        assert_eq!(comm1.total_weight, 11.0);
        assert!(comm1.vertices.contains(&VertexID(0)));

        assert!(state_manager.updated_communities.contains(&CommunityID(1)));

        // Remove vertex 0 from community 1
        let remove_movement = VertexMovement {
            community_id: CommunityID(1),
            vtx: VertexID(0),
            internal_weight: 1.0,
            total_weight: 3.0,
            operation: CommunityOperation::Remove,
        };
        state_manager.move_vertex(&remove_movement);

        let comm1_after_remove = state_manager
            .tracked_communities
            .get(&CommunityID(1))
            .unwrap();
        assert_eq!(comm1_after_remove.internal_weight, 0.0);
        assert_eq!(comm1_after_remove.total_weight, 8.0);
        assert!(!comm1_after_remove.vertices.contains(&VertexID(0)));
    }

    #[test]
    fn test_get_updated_communities() {
        let graph = create_test_graph();
        let mut state_manager = StateManager::new(&graph);

        // Update multiple communities
        let movements = vec![
            VertexMovement {
                community_id: CommunityID(1),
                vtx: VertexID(0),
                internal_weight: 1.0,
                total_weight: 3.0,
                operation: CommunityOperation::Insert,
            },
            VertexMovement {
                community_id: CommunityID(2),
                vtx: VertexID(1),
                internal_weight: 3.0,
                total_weight: 7.0,
                operation: CommunityOperation::Insert,
            },
        ];

        for movement in movements {
            state_manager.move_vertex(&movement);
        }

        let updated: Vec<_> = state_manager.get_updated_communities().collect();
        assert_eq!(updated.len(), 2);
        assert!(updated.contains(&&CommunityID(1)));
        assert!(updated.contains(&&CommunityID(2)));
    }

    #[test]
    fn test_get_local_community() {
        let graph = create_test_graph();
        let state_manager = StateManager::new(&graph);

        for i in 0..5 {
            let local_comm = state_manager.get_local_community(&CommunityID(i));
            assert_eq!(local_comm.id, CommunityID(i));
            assert!(local_comm.is_local);
        }

        // Test for non-existent community
        let result = std::panic::catch_unwind(|| {
            state_manager.get_local_community(&CommunityID(10));
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_local_communities() {
        let graph = create_test_graph();
        let state_manager = StateManager::new(&graph);

        let local_comms: Vec<_> = state_manager.local_communities().collect();
        assert_eq!(
            local_comms,
            vec![
                &CommunityID(0),
                &CommunityID(1),
                &CommunityID(2),
                &CommunityID(3),
                &CommunityID(4)
            ]
        );
    }

    #[test]
    fn test_community_of() {
        let graph = create_test_graph();
        let state_manager = StateManager::new(&graph);

        for i in 0..5 {
            assert_eq!(state_manager.community_of(&VertexID(i)), &CommunityID(i));
        }

        // Test for non-existent vertex
        let result = std::panic::catch_unwind(|| {
            state_manager.community_of(&VertexID(10));
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_process_vertex_movements() {
        let graph = create_test_graph();
        let mut state_manager = StateManager::new(&graph);

        let movements = vec![
            // remove vtx 0 from community 0
            VertexMovement {
                community_id: CommunityID(0),
                vtx: VertexID(0),
                internal_weight: 0.0,
                total_weight: graph.weighted_degree(0usize.into()),
                operation: CommunityOperation::Remove,
            },
            // move vtx 0 to community 1
            VertexMovement {
                community_id: CommunityID(1),
                vtx: VertexID(0),
                internal_weight: 1.0,
                total_weight: graph.weighted_degree(0usize.into()),
                operation: CommunityOperation::Insert,
            },
            // remove vtx 1 from community 1
            VertexMovement {
                community_id: CommunityID(1),
                vtx: VertexID(1),
                // internal weight of 1.0 simulates the update that occurs when
                // 0 joins this community
                internal_weight: 1.0,
                total_weight: graph.weighted_degree(1usize.into()),
                operation: CommunityOperation::Remove,
            },
            // move vtx 1 to community 2
            VertexMovement {
                community_id: CommunityID(2),
                vtx: VertexID(1),
                internal_weight: 3.0,
                total_weight: graph.weighted_degree(1usize.into()),
                operation: CommunityOperation::Insert,
            },
        ];

        state_manager.batch_process_vertex_movements(movements.iter());

        assert_eq!(
            state_manager.vtx_community_map.get(&VertexID(0)),
            Some(&CommunityID(1))
        );
        assert_eq!(
            state_manager.vtx_community_map.get(&VertexID(1)),
            Some(&CommunityID(2))
        );

        let comm0 = state_manager
            .tracked_communities
            .get(&CommunityID(0))
            .unwrap();

        assert!(comm0.vertices.is_empty());
        assert_eq!(comm0.internal_weight, 0.0);
        assert_eq!(comm0.total_weight, 0.0);

        let comm1 = state_manager
            .tracked_communities
            .get(&CommunityID(1))
            .unwrap();
        assert_eq!(comm1.vertices.len(), 1);
        assert_eq!(comm1.internal_weight, 0.0);
        assert_eq!(comm1.total_weight, 3.0);
        assert!(comm1.vertices.contains(&VertexID(0)));

        let comm2 = state_manager
            .tracked_communities
            .get(&CommunityID(2))
            .unwrap();
        assert_eq!(comm2.internal_weight, 6.0);
        assert_eq!(comm2.total_weight, 18.0);

        assert!(state_manager.updated_communities.contains(&CommunityID(0)));
        assert!(state_manager.updated_communities.contains(&CommunityID(1)));
        assert!(state_manager.updated_communities.contains(&CommunityID(2)));
    }

    #[test]
    fn test_batch_process_community_update() {
        let graph = create_test_graph();
        let mut state_manager = StateManager::new(&graph);

        // mock vector of received community updates
        let updates = vec![
            CommunityInfo {
                id: CommunityID(0),
                internal_weight: 1.5,
                total_weight: 3.0,
            },
            CommunityInfo {
                id: CommunityID(10),
                internal_weight: 2.0,
                total_weight: 4.0,
            },
        ];

        state_manager.batch_process_community_update(updates.iter());

        let updated_comm0 = state_manager
            .tracked_communities
            .get(&CommunityID(0))
            .unwrap();
        assert_eq!(updated_comm0.internal_weight, 1.5);
        assert_eq!(updated_comm0.total_weight, 3.0);

        let new_comm10 = state_manager
            .tracked_communities
            .get(&CommunityID(10))
            .unwrap();
        assert_eq!(new_comm10.internal_weight, 2.0);
        assert_eq!(new_comm10.total_weight, 4.0);
        assert!(!new_comm10.is_local);
    }

    #[test]
    fn test_batch_process_neighbor_info() {
        let graph = create_test_graph();
        let mut state_manager = StateManager::new(&graph);

        let neighbor_info = vec![
            NeighborInfo {
                vertex_id: VertexID(0),
                community_id: CommunityID(1),
            },
            NeighborInfo {
                vertex_id: VertexID(1),
                community_id: CommunityID(2),
            },
        ];

        state_manager.batch_process_neighor_info(neighbor_info.iter());

        assert_eq!(
            state_manager.vtx_community_map.get(&VertexID(0)),
            Some(&CommunityID(1))
        );
        assert_eq!(
            state_manager.vtx_community_map.get(&VertexID(1)),
            Some(&CommunityID(2))
        );
    }

    #[test]
    fn test_large_graph() {
        // logger::init(0);

        let graph = create_large_test_graph();
        let mut state_manager = StateManager::new(&graph);

        assert_eq!(state_manager.tracked_communities.len(), 100);
        assert_eq!(state_manager.vtx_community_map.len(), 100);

        // Move some vertices
        let movements = (0..50)
            .map(|i| {
                [
                    VertexMovement {
                        community_id: CommunityID(i),
                        vtx: VertexID(i),
                        internal_weight: 0.0,
                        total_weight: graph.weighted_degree(i),
                        operation: CommunityOperation::Remove,
                    },
                    VertexMovement {
                        community_id: CommunityID(i / 10),
                        vtx: VertexID(i),
                        internal_weight: i as f64,
                        total_weight: graph.weighted_degree(i),
                        operation: CommunityOperation::Insert,
                    },
                ]
            })
            .flatten()
            .collect::<Vec<_>>();

        state_manager.batch_process_vertex_movements(movements.iter());

        // Check if vertices were moved correctly
        for i in 0..50 {
            assert_eq!(
                state_manager.vtx_community_map.get(&VertexID(i)),
                Some(&CommunityID(i / 10))
            );
        }

        // Check if communities were updated correctly
        for i in 0..5 {
            let comm = state_manager
                .tracked_communities
                .get(&CommunityID(i))
                .unwrap();
            assert!(comm.internal_weight > 0.0);
            assert!(comm.total_weight > 0.0);
            assert!(comm.vertices.len() > 1);
        }

        for i in 5..50 {
            let comm = state_manager
                .tracked_communities
                .get(&CommunityID(i))
                .unwrap();

            // total weight and internal weight should be 0
            assert!(f64::abs(0.0 - comm.internal_weight) < 0.00001);
            assert!(f64::abs(0.0 - comm.total_weight) < 0.00001);
            assert!(comm.vertices.is_empty());
        }

        // Check updated communities
        let updated: Vec<_> = state_manager.get_updated_communities().collect();
        debug!("{:#?}", updated);
        debug!(
            "{:#?}",
            state_manager
                .tracked_communities
                .iter()
                .take(50)
                .map(|(_, comm)| comm)
                .collect::<Vec<&Community>>()
        );
        assert_eq!(updated.len(), 50);
    }
}
