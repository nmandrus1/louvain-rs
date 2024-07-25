use indexmap::IndexMap;
use mpi::{
    request::{LocalScope, RequestCollection},
    traits::{Destination, Equivalence, Source},
    Count, Rank,
};

use crate::CommunityID;

use super::{MessageManagerTrait, MessageRouter};

pub enum LouvainMessage<'u> {
    CommunityUpdate(&'u CommunityUpdate),
    VertexMovement(&'u VertexMovement),
    NeighborInfo(&'u NeighborInfo),
}

/// When a community's state changes, send an update to all processes with
/// a reference to this community
#[derive(Default, Equivalence, Clone, Copy, Debug, PartialEq)]
pub struct CommunityUpdate {
    id: CommunityID,
    internal_weight: f64,
    total_weight: f64,
}

/// Message to send to a remote process when a vertex is moving
#[derive(Default, Clone, Copy, Equivalence, Debug, PartialEq)]
pub struct VertexMovement {
    // target of movement
    new_community: CommunityID,
    vtx: usize,
    internal_weight: f64,
    total_weight: f64,
}

/// Message to send to remote processes with neighbors to this vertex
/// to ensure they have the most up-to-date state when processing their
/// next vertices
#[derive(Default, Equivalence, Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeighborInfo {
    vertex_id: usize,
    community_id: CommunityID,
}

pub struct MessageManager<'r, R: MessageRouter> {
    router: &'r R,

    // Goal: Queue up messages to all ranks into a buffer and send the whole buffer
    // so that for each message we can perform a single send/recv call for each rank.

    // Goal 2: Allreduce to gather the expected number of messages from each rank and
    // allocate the proper amount in our receive buffers, and post the receives so that
    // they can all be filled asynchronously

    // Buffers for sending messages
    community_updates: IndexMap<Rank, Vec<CommunityUpdate>>,
    vertex_movements: IndexMap<Rank, Vec<VertexMovement>>,
    neighbor_info: IndexMap<Rank, Vec<NeighborInfo>>,

    // Buffers for receiving messages
    // recv_community_updates: IndexMap<Rank, RefCell<Vec<CommunityUpdate>>>,
    // recv_vertex_movements: IndexMap<Rank, RefCell<Vec<VertexMovement>>>,
    // recv_neighbor_info: IndexMap<Rank, RefCell<Vec<NeighborInfo>>>,
    recv_community_updates: IndexMap<Rank, Vec<CommunityUpdate>>,
    recv_vertex_movements: IndexMap<Rank, Vec<VertexMovement>>,
    recv_neighbor_info: IndexMap<Rank, Vec<NeighborInfo>>,

    // Vector with an entry for each rank containing the number of
    // elements that are being sent/recvd to/from the corresponding rank
    send_counts: Vec<Count>,
    recv_counts: Vec<Count>,
}

impl<'r, R: MessageRouter> MessageManager<'r, R> {
    const COMMUNITY_UPDATE: i32 = 10;
    const VERTEX_MOVEMNT: i32 = 10;
    const NEIGHBOR_INFO: i32 = 10;

    pub fn new(router: &'r R) -> Self {
        // Initialize the struct
        let size = router.size();

        Self {
            router,

            community_updates: IndexMap::with_capacity(size),

            vertex_movements: IndexMap::with_capacity(size),
            neighbor_info: IndexMap::with_capacity(size),

            recv_community_updates: IndexMap::with_capacity(size),
            recv_vertex_movements: IndexMap::with_capacity(size),
            recv_neighbor_info: IndexMap::with_capacity(size),

            send_counts: vec![0; size],
            recv_counts: vec![0; size],
        }
    }

    /// communicate global modularity
    pub fn global_modularity(&self, partial: f64) -> f64 {
        self.router.global_modularity(partial)
    }

    fn clear_counts(&mut self) {
        self.send_counts.iter_mut().for_each(|c| *c = 0);
        self.recv_counts.iter_mut().for_each(|c| *c = 0);
    }

    fn clear(&mut self) {
        self.community_updates.values_mut().for_each(|v| v.clear());

        self.vertex_movements.values_mut().for_each(|v| v.clear());

        self.neighbor_info.values_mut().for_each(|v| v.clear());

        self.recv_community_updates
            .values_mut()
            .for_each(|v| v.clear());

        self.recv_vertex_movements
            .values_mut()
            .for_each(|v| v.clear());
        self.recv_neighbor_info.values_mut().for_each(|v| v.clear());

        self.clear_counts();
    }

    // non blocking send calls with requests stored for waiting
    fn send_buffer_async_with_tag<'buf, 'scope, D: Equivalence>(
        &'buf self,
        scope: &LocalScope<'scope>,
        send_buf: &'buf IndexMap<Rank, Vec<D>>,
        tag: i32,
        coll: &mut RequestCollection<'scope, Vec<D>>,
    ) where
        'buf: 'scope,
    {
        send_buf.iter().for_each(|(rank, buf)| {
            let sreq = self
                .router
                .process_at_rank(*rank)
                .immediate_send_with_tag(scope, buf, tag);

            coll.add(sreq);
        });
    }

    fn receive_async_to_buf<'scope, 'buf, D: Equivalence>(
        &self,
        scope: &LocalScope<'scope>,
        recv_buf: &'buf mut IndexMap<Rank, Vec<D>>,
        coll: &mut RequestCollection<'scope, Vec<D>>,
    ) where
        'buf: 'scope,
    {
        // post receives based on the earlier collective communication
        // to signal who is receiving messages
        recv_buf.iter_mut().for_each(|(source, buf)| {
            let rreq = self
                .router
                .process_at_rank(*source)
                // TODO: Maybe match the tag?
                .immediate_receive_into(scope, buf);
            coll.add(rreq);
        })
    }

    // Helper Methods

    /// Gets the counts from the send_map, and does an all_to_all communication step to
    /// aggregate how much data we should expect from each participating process
    /// sets the recv_counts field
    fn get_send_and_recv_counts<V>(
        &mut self,
        send_map: &IndexMap<Rank, Vec<V>>,
    ) -> anyhow::Result<(usize, usize)> {
        self.clear_counts();

        send_map
            .iter()
            .for_each(|(rank, buf)| self.send_counts[*rank as usize] = buf.len() as Count);

        self.router
            .distribute_counts(&self.send_counts, &mut self.recv_counts)?;

        let send_total = self.send_counts.iter().sum::<i32>().try_into()?;
        let recv_total = self.recv_counts.iter().sum::<i32>().try_into()?;
        Ok((send_total, recv_total))
    }

    fn allocate_recv_buf<V: Default>(
        &self,
        recv_buf: &mut IndexMap<Rank, Vec<V>>,
        recv_counts: &[Count],
    ) {
        // TODO: Is reserving space in the vec correct? Or must it be filled with default entries?
        recv_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| count.is_positive())
            .for_each(|(idx, count)| {
                recv_buf
                    .entry(idx as Rank)
                    .or_default()
                    .resize_with(*count as usize, || V::default())
            })
    }

    /// Send vertex movements and
    fn exchange_buffer_with_tag<V: Default + Equivalence>(
        &mut self,
        send_buf: &IndexMap<Rank, Vec<V>>,
        recv_buf: &mut IndexMap<Rank, Vec<V>>,
        tag: i32,
    ) -> anyhow::Result<()> {
        // Receive buffers should be empty!!
        debug_assert!(recv_buf.values().all(|vec| vec.is_empty()));

        let (send_count_total, recv_count_total) = self.get_send_and_recv_counts(&send_buf)?;
        self.allocate_recv_buf(recv_buf, &self.recv_counts);

        let this_proc = self.router.this_process();

        // this is used to reserve space for all the requests we will process as
        // a result of this communication step
        let request_count = send_count_total + recv_count_total;

        mpi::request::multiple_scope(request_count, |scope, coll| {
            // post send/recv calls
            self.send_buffer_async_with_tag(scope, send_buf, tag, coll);
            self.receive_async_to_buf(scope, recv_buf, coll);

            let mut send_count = 0;
            let mut recv_count = 0;
            while coll.incomplete() > 0 {
                let (_, status, result) = coll.wait_any().unwrap();
                if status.source_rank() == this_proc.rank() {
                    send_count += result.len();
                } else {
                    recv_count += result.len();
                }
            }

            debug_assert_eq!(send_count, send_count_total);
            debug_assert_eq!(recv_count, recv_count_total);
        });

        Ok(())
    }
}

impl<'r, R: MessageRouter> MessageManagerTrait for MessageManager<'r, R> {
    /// Queue community update to send later
    fn queue_community_update(&mut self, update: CommunityUpdate) {
        self.router
            .route(LouvainMessage::CommunityUpdate(&update))
            .into_iter()
            .for_each(|rank| {
                self.community_updates.entry(rank).or_default().push(update);
            });
    }

    /// Queue a vertex movement to be sent later
    fn queue_vertex_movement(&mut self, movement: VertexMovement) {
        self.router
            .route(LouvainMessage::VertexMovement(&movement))
            .into_iter()
            .for_each(|rank| {
                self.vertex_movements
                    .entry(rank)
                    .or_default()
                    .push(movement)
            })
    }

    fn queue_neighbor_info(&mut self, info: NeighborInfo) {
        self.router
            .route(LouvainMessage::NeighborInfo(&info))
            .into_iter()
            .for_each(|rank| self.neighbor_info.entry(rank).or_default().push(info))
    }

    /// Send vertex movements and
    fn exchange_vertex_movements(&mut self) -> anyhow::Result<()> {
        // Receive buffers should be empty!!
        debug_assert!(self
            .recv_vertex_movements
            .values()
            .all(|vec| vec.is_empty()));

        let send_buf = std::mem::take(&mut self.vertex_movements);
        let mut recv_buf = std::mem::take(&mut self.recv_vertex_movements);

        self.exchange_buffer_with_tag(&send_buf, &mut recv_buf, Self::VERTEX_MOVEMNT)?;

        // replace stolen values with the filled buffers
        self.vertex_movements = send_buf;
        self.recv_vertex_movements = recv_buf;

        Ok(())
    }

    fn exchange_other_messages(&mut self) -> anyhow::Result<()> {
        // Receive buffers should be empty!!
        debug_assert!(self
            .recv_community_updates
            .values()
            .all(|vec| vec.is_empty()));

        debug_assert!(self.recv_neighbor_info.values().all(|vec| vec.is_empty()));

        let community_update_send_buf = std::mem::take(&mut self.community_updates);
        let mut community_update_recv_buf = std::mem::take(&mut self.recv_community_updates);
        let neighbor_info_send_buf = std::mem::take(&mut self.neighbor_info);
        let mut neighbor_info_recv_buf = std::mem::take(&mut self.recv_neighbor_info);

        self.exchange_buffer_with_tag(
            &community_update_send_buf,
            &mut community_update_recv_buf,
            Self::COMMUNITY_UPDATE,
        )?;

        self.exchange_buffer_with_tag(
            &neighbor_info_send_buf,
            &mut neighbor_info_recv_buf,
            Self::NEIGHBOR_INFO,
        )?;

        // replace stolen values with the filled buffers
        self.community_updates = community_update_send_buf;
        self.recv_community_updates = community_update_recv_buf;
        self.neighbor_info = neighbor_info_send_buf;
        self.recv_neighbor_info = neighbor_info_recv_buf;

        Ok(())
    }

    fn clear(&mut self) {
        self.clear();
    }

    fn get_received_community_updates(&self) -> impl Iterator<Item = &CommunityUpdate> {
        self.recv_community_updates.values().flatten()
    }

    fn get_received_vertex_movements(&self) -> impl Iterator<Item = &VertexMovement> {
        self.recv_vertex_movements.values().flatten()
    }

    fn get_received_neighbor_info(&self) -> impl Iterator<Item = &NeighborInfo> {
        self.recv_neighbor_info.values().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpi::topology::Process;

    // Mock implementation of MessageRouter
    struct MockMessageRouter {
        size: usize,
        _rank: Rank,
    }

    impl MessageRouter for MockMessageRouter {
        fn size(&self) -> usize {
            self.size
        }

        fn route(&self, message: LouvainMessage) -> Vec<Rank> {
            match message {
                LouvainMessage::CommunityUpdate(_) => vec![0, 1],
                LouvainMessage::VertexMovement(_) => vec![1, 2],
                LouvainMessage::NeighborInfo(_) => vec![0, 2],
            }
        }

        fn process_at_rank(&self, _rank: Rank) -> Process {
            unimplemented!("This is a mock implementation")
        }

        fn this_process(&self) -> Process {
            unimplemented!("This is a mock implementation")
        }

        fn distribute_counts(
            &self,
            _send_counts: &[Count],
            recv_counts: &mut [Count],
        ) -> anyhow::Result<()> {
            // Mock implementation: set recv_counts to some predefined values
            recv_counts[0] = 2;
            recv_counts[1] = 1;
            recv_counts[2] = 3;
            Ok(())
        }

        fn global_modularity(&self, _partial: f64) -> f64 {
            unimplemented!("This is a mock implemenation")
        }
    }

    #[test]
    fn test_queue_community_update() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let mut manager = MessageManager::new(&router);

        let update = CommunityUpdate {
            id: CommunityID(1),
            internal_weight: 2.0,
            total_weight: 5.0,
        };

        manager.queue_community_update(update);

        assert_eq!(manager.community_updates.len(), 2);
        assert!(manager.community_updates.contains_key(&0));
        assert!(manager.community_updates.contains_key(&1));
        assert_eq!(manager.community_updates[&0].len(), 1);
        assert_eq!(manager.community_updates[&1].len(), 1);
        assert_eq!(manager.community_updates[&0][0], update);
        assert_eq!(manager.community_updates[&1][0], update);
    }

    #[test]
    fn test_queue_vertex_movement() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let mut manager = MessageManager::new(&router);

        let movement = VertexMovement {
            new_community: CommunityID(2),
            vtx: 5,
            internal_weight: 1.5,
            total_weight: 3.0,
        };

        manager.queue_vertex_movement(movement);

        assert_eq!(manager.vertex_movements.len(), 2);
        assert!(manager.vertex_movements.contains_key(&1));
        assert!(manager.vertex_movements.contains_key(&2));
        assert_eq!(manager.vertex_movements[&1].len(), 1);
        assert_eq!(manager.vertex_movements[&2].len(), 1);
        assert_eq!(manager.vertex_movements[&1][0], movement);
        assert_eq!(manager.vertex_movements[&2][0], movement);
    }

    #[test]
    fn test_queue_neighbor_info() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let mut manager = MessageManager::new(&router);

        let info = NeighborInfo {
            vertex_id: 3,
            community_id: CommunityID(4),
        };

        manager.queue_neighbor_info(info);

        assert_eq!(manager.neighbor_info.len(), 2);
        assert!(manager.neighbor_info.contains_key(&0));
        assert!(manager.neighbor_info.contains_key(&2));
        assert_eq!(manager.neighbor_info[&0].len(), 1);
        assert_eq!(manager.neighbor_info[&2].len(), 1);
        assert_eq!(manager.neighbor_info[&0][0], info);
        assert_eq!(manager.neighbor_info[&2][0], info);
    }

    #[test]
    fn test_clear() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let mut manager = MessageManager::new(&router);

        // Queue some messages
        manager.queue_community_update(CommunityUpdate::default());
        manager.queue_vertex_movement(VertexMovement::default());
        manager.queue_neighbor_info(NeighborInfo::default());

        // Simulate receiving some messages
        manager
            .recv_community_updates
            .insert(0, vec![CommunityUpdate::default()]);
        manager
            .recv_vertex_movements
            .insert(1, vec![VertexMovement::default()]);
        manager
            .recv_neighbor_info
            .insert(2, vec![NeighborInfo::default()]);

        // Set some counts
        manager.send_counts = vec![1, 2, 3];
        manager.recv_counts = vec![4, 5, 6];

        // Clear everything
        manager.clear();

        // Check if everything is cleared
        assert!(manager.community_updates.values().all(|v| v.is_empty()));
        assert!(manager.vertex_movements.values().all(|v| v.is_empty()));
        assert!(manager.neighbor_info.values().all(|v| v.is_empty()));
        assert!(manager
            .recv_community_updates
            .values()
            .all(|v| v.is_empty()));
        assert!(manager.recv_vertex_movements.values().all(|v| v.is_empty()));
        assert!(manager.recv_neighbor_info.values().all(|v| v.is_empty()));
        assert!(manager.send_counts.iter().all(|&c| c == 0));
        assert!(manager.recv_counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_get_send_and_recv_counts() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let mut manager = MessageManager::new(&router);

        let mut send_map = IndexMap::new();
        send_map.insert(0, vec![1, 2, 3]);
        send_map.insert(1, vec![4]);
        send_map.insert(2, vec![5, 6]);

        let (send_total, recv_total) = manager.get_send_and_recv_counts(&send_map).unwrap();

        assert_eq!(send_total, 6);
        assert_eq!(recv_total, 6);
        assert_eq!(manager.send_counts, vec![3, 1, 2]);
        assert_eq!(manager.recv_counts, vec![2, 1, 3]);
    }

    #[test]
    fn test_allocate_recv_buf() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let manager = MessageManager::new(&router);

        let mut recv_buf = IndexMap::new();
        let recv_counts = vec![2, 0, 3];

        manager.allocate_recv_buf::<NeighborInfo>(&mut recv_buf, &recv_counts);

        assert_eq!(recv_buf.len(), 2);
        assert_eq!(recv_buf[&0].len(), 2);
        assert_eq!(recv_buf[&2].len(), 3);
        assert!(!recv_buf.contains_key(&1));
    }

    // #[test]
    // fn test_exchange_buffer_with_tag() {
    //     let router = MockMessageRouter { size: 3, _rank: 1 };
    //     let mut manager = MessageManager::new(&router);

    //     let mut send_buf = IndexMap::new();
    //     send_buf.insert(0, vec![CommunityUpdate::default(); 2]);
    //     send_buf.insert(2, vec![CommunityUpdate::default(); 3]);

    //     let mut recv_buf = IndexMap::new();

    //     manager
    //         .exchange_buffer_with_tag(&send_buf, &mut recv_buf, 10)
    //         .unwrap();

    //     assert_eq!(recv_buf.len(), 3);
    //     assert_eq!(recv_buf[&0].len(), 2);
    //     assert_eq!(recv_buf[&1].len(), 1);
    //     assert_eq!(recv_buf[&2].len(), 3);
    // }

    // #[test]
    // fn test_exchange_vertex_movements() {
    //     let router = MockMessageRouter { size: 3, _rank: 1 };
    //     let mut manager = MessageManager::new(&router);

    //     manager
    //         .vertex_movements
    //         .insert(0, vec![VertexMovement::default(); 2]);
    //     manager
    //         .vertex_movements
    //         .insert(2, vec![VertexMovement::default(); 3]);

    //     manager.exchange_vertex_movements().unwrap();

    //     assert_eq!(manager.recv_vertex_movements.len(), 3);
    //     assert_eq!(manager.recv_vertex_movements[&0].len(), 2);
    //     assert_eq!(manager.recv_vertex_movements[&1].len(), 1);
    //     assert_eq!(manager.recv_vertex_movements[&2].len(), 3);
    // }

    // #[test]
    // fn test_exchange_other_messages() {
    //     let router = MockMessageRouter { size: 3, _rank: 1 };
    //     let mut manager = MessageManager::new(&router);

    //     manager
    //         .community_updates
    //         .insert(0, vec![CommunityUpdate::default(); 2]);
    //     manager
    //         .neighbor_info
    //         .insert(2, vec![NeighborInfo::default(); 3]);

    //     manager.exchange_other_messages().unwrap();

    //     assert_eq!(manager.recv_community_updates.len(), 3);
    //     assert_eq!(manager.recv_community_updates[&0].len(), 2);
    //     assert_eq!(manager.recv_community_updates[&1].len(), 1);
    //     assert_eq!(manager.recv_community_updates[&2].len(), 3);

    //     assert_eq!(manager.recv_neighbor_info.len(), 3);
    //     assert_eq!(manager.recv_neighbor_info[&0].len(), 2);
    //     assert_eq!(manager.recv_neighbor_info[&1].len(), 1);
    //     assert_eq!(manager.recv_neighbor_info[&2].len(), 3);
    // }

    #[test]
    fn test_get_received_messages() {
        let router = MockMessageRouter { size: 3, _rank: 1 };
        let mut manager = MessageManager::new(&router);

        let community_update = CommunityUpdate {
            id: CommunityID(1),
            internal_weight: 2.0,
            total_weight: 5.0,
        };
        let vertex_movement = VertexMovement {
            new_community: CommunityID(2),
            vtx: 5,
            internal_weight: 1.5,
            total_weight: 3.0,
        };
        let neighbor_info = NeighborInfo {
            vertex_id: 3,
            community_id: CommunityID(4),
        };

        manager
            .recv_community_updates
            .insert(0, vec![community_update]);
        manager
            .recv_vertex_movements
            .insert(1, vec![vertex_movement]);

        // insert 4 updates
        manager.recv_neighbor_info.insert(2, vec![neighbor_info]);
        manager.recv_neighbor_info.insert(3, vec![neighbor_info; 3]);

        let received_community_updates: Vec<&CommunityUpdate> =
            manager.get_received_community_updates().collect();
        let received_vertex_movements: Vec<&VertexMovement> =
            manager.get_received_vertex_movements().collect();
        let received_neighbor_info: Vec<&NeighborInfo> =
            manager.get_received_neighbor_info().collect();

        assert_eq!(received_community_updates.len(), 1);
        assert_eq!(received_community_updates[0], &community_update);

        assert_eq!(received_vertex_movements.len(), 1);
        assert_eq!(received_vertex_movements[0], &vertex_movement);

        assert_eq!(received_neighbor_info.len(), 4);
        assert_eq!(received_neighbor_info[0], &neighbor_info);
        assert_eq!(received_neighbor_info[1], &neighbor_info);
        assert_eq!(received_neighbor_info[2], &neighbor_info);
        assert_eq!(received_neighbor_info[3], &neighbor_info);
    }
}
