use crate::{MessageManagerTrait, MessageRouter, VertexID};

use super::{
    CommunityInfo, CommunityState, LouvainGraph, MessageManager, NeighborInfo, VertexMovement,
};

use mpi::traits::Equivalence;

use log::{debug, info};

pub enum IterationStep {
    Process(VertexID),
    SyncOnly,
}

/// struct responsible for carrying out distributed Louvain Method
pub struct DistributedCommunityDetection<'a, G: LouvainGraph, R: MessageRouter, S: CommunityState> {
    graph: &'a G,
    message_manager: MessageManager<'a, R>,
    state: S,
}

impl<'a, G: LouvainGraph, R: MessageRouter, S: CommunityState>
    DistributedCommunityDetection<'a, G, R, S>
{
    pub fn new(graph: &'a G, router: &'a R, state: S) -> Self {
        info!("Creating new DistributedCommunityDetection instance");

        Self {
            graph,
            state,
            message_manager: MessageManager::new(router),
        }
    }

    pub fn one_level(&mut self) {
        info!("Starting one_level of community detection");
        self.iterate();

        info!("Completed one_level of community detection");
    }

    fn local_modularity(&self) -> f64 {
        debug!("Calculating local modularity");
        let m2 = self.graph.global_edge_count() as f64 * 2.0;

        let result = self
            .state
            .local_communities()
            .map(|id| self.state.get_local_community(id))
            .filter_map(|comm| {
                (comm.total_weight > 0.0)
                    .then_some((comm.internal_weight / m2) - (comm.total_weight / m2).powi(2))
            })
            .sum();

        debug!("Local modularity calculated: {}", result);
        result
    }

    pub fn global_modularity(&self) -> f64 {
        info!("Calculating global modularity");
        let partial = self.local_modularity();
        let result = self.message_manager.global_modularity(partial);
        result
    }

    fn iterate(&mut self) {
        self.graph.vertex_feed().for_each(|step| {
            let processed_vertex = match step {
                IterationStep::Process(vtx) => {
                    // returns Some(vtx) if the vertex was moved
                    // None if it stayed in its community
                    todo!()
                }

                // move on to sync stage
                IterationStep::SyncOnly => None,
            };

            self.syncrhonize(processed_vertex);
        });
    }

    fn syncrhonize(&mut self, processed_vertex: Option<VertexID>) -> anyhow::Result<()> {
        {
            let state = &mut self.state;
            let message_manager = &mut self.message_manager;

            message_manager.exchange_vertex_movements()?;
            let movements = self.message_manager.get_received_vertex_movements();
            Self::process_movements(movements, state);
        }

        {
            let state = &mut self.state;
            let message_manager = &mut self.message_manager;
            // queue and buffer all messages
            Self::queue_community_updates(message_manager, state);

            if let Some(vtx) = processed_vertex {
                let info = NeighborInfo {
                    vertex_id: vtx,
                    community_id: *state.community_of(&vtx),
                };
                Self::queue_vertex_neighbor_updates(info, message_manager, state);
            }

            // send messages
            message_manager.exchange_other_messages()?;
            let updates = message_manager.get_received_community_updates();
            let info = message_manager.get_received_neighbor_info();

            // process incoming updates
            Self::process_community_updates(updates, state);
            Self::process_neighbor_info(info, state);
        }

        Ok(())
    }

    /// for all communities that have been updates, send out the
    /// updated info to all the necessary ranks
    #[inline]
    fn queue_community_updates(manager: &mut MessageManager<'a, R>, state: &S) {
        // TODO: test this function
        state.get_updated_communities().for_each(|comm_id| {
            manager.queue_community_update(state.get_local_community(&comm_id).get_info(), state)
        });
    }

    // TODO: Test
    #[inline]
    fn queue_vertex_neighbor_updates(
        info: NeighborInfo,
        manager: &mut MessageManager<'a, R>,
        state: &S,
    ) {
        manager.queue_neighbor_info(info, state)
    }

    /// processes all of the received movements from the message manager,
    /// and populates the vector of updated communities
    #[inline]
    fn process_movements<'recv>(
        movements: impl Iterator<Item = &'recv VertexMovement>,
        state: &mut S,
    ) {
        state.batch_process_vertex_movements(movements);
    }

    // Receive info from remote processes about communities we are keeping track of
    #[inline]
    fn process_community_updates<'recv>(
        updates: impl Iterator<Item = &'recv CommunityInfo>,
        state: &mut S,
    ) {
        state.batch_process_community_update(updates)
    }

    // Receive info from remote processes about vertices whose communities have changed
    #[inline]
    fn process_neighbor_info<'recv>(
        info: impl Iterator<Item = &'recv NeighborInfo>,
        state: &mut S,
    ) {
        state.batch_process_neighor_info(info)
    }
}

#[cfg(test)]
mod test {

    // test initialization
    // test neighbor computation
    // test best community
    // test vertex insertion/removal
}
