use distributed_louvain::*;
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives},
};
use petgraph::{csr::Csr, visit::IntoNeighbors};

/// Given the size of the MPI topology, and the global number of vertices, compute the row
/// ownership of every rank
fn compute_edgelist_distribution(size: usize, edges: &[Edge]) -> Vec<(usize, usize)> {
    let mut edge_partitions = Vec::with_capacity(size);
    edge_partitions.resize(size, (0, 0));

    let edges_per_proc = edges.len() / size;
    let leftover_edges = edges.len() % size;

    for (rank, rows) in edge_partitions.iter_mut().enumerate() {
        let leftover = if rank < leftover_edges { 1 } else { 0 };
        rows.0 = rank * edges_per_proc + std::cmp::min(rank, leftover_edges);
        rows.1 = rows.0 + edges_per_proc + leftover;
    }

    edge_partitions
}

fn main() -> anyhow::Result<()> {
    // integration test for small edge list

    let small_edge_list: [Edge; 28] = [
        Edge::from((1, 2, 1.0)),
        Edge::from((1, 4, 1.0)),
        Edge::from((1, 7, 1.0)),
        Edge::from((2, 0, 1.0)),
        Edge::from((2, 4, 1.0)),
        Edge::from((2, 5, 1.0)),
        Edge::from((2, 6, 1.0)),
        Edge::from((3, 0, 1.0)),
        Edge::from((3, 7, 1.0)),
        Edge::from((4, 0, 1.0)),
        Edge::from((4, 10, 1.0)),
        Edge::from((5, 0, 1.0)),
        Edge::from((5, 7, 1.0)),
        Edge::from((5, 11, 1.0)),
        Edge::from((6, 7, 1.0)),
        Edge::from((6, 11, 1.0)),
        Edge::from((8, 9, 1.0)),
        Edge::from((8, 10, 1.0)),
        Edge::from((8, 11, 1.0)),
        Edge::from((8, 14, 1.0)),
        Edge::from((8, 15, 1.0)),
        Edge::from((9, 12, 1.0)),
        Edge::from((9, 14, 1.0)),
        Edge::from((10, 11, 1.0)),
        Edge::from((10, 12, 1.0)),
        Edge::from((10, 13, 1.0)),
        Edge::from((10, 14, 1.0)),
        Edge::from((11, 13, 1.0)),
    ];

    let (_universe, world) = distributed_louvain::init()?;
    let size = world.size() as usize;
    let rank = world.rank() as usize;

    // 16 vertices in small graph
    let edge_ownership = compute_edgelist_distribution(size, &small_edge_list);

    // tuple of indeces of the edge list that this process owns
    let edges = edge_ownership[rank];

    // partition edge list based on rank to simulate distributed loading
    let edges = &small_edge_list[edges.0..edges.1];

    let dg = DistributedGraph::from_distributed(edges, &world)?;

    let mut csr = Csr::<usize, f64, petgraph::Undirected, usize>::with_nodes(16);
    for edge in small_edge_list {
        csr.add_edge(edge.0 .0, edge.1 .0, edge.2);
    }

    assert_eq!(dg.info.global_ecount(), 56);
    assert_eq!(dg.info.global_vcount(), 16);

    for vtx in dg.vertices() {
        if rank == dg.owner_of_vertex(vtx) as usize {
            assert!(dg
                .neighbors(vtx)
                .map(|(neigh, _)| neigh.0)
                .eq(csr.neighbors(vtx.0)));
        }
    }

    if rank == 0 {
        println!("Integration Test Passed")
    }

    let buf;

    if rank == 1 {
        buf = [100, 0, 0, 0];
    } else {
        buf = [1, 2, 3, 4];
    }

    let mut count = [0];
    world.reduce_scatter_block_into(&buf, &mut count, SystemOperation::sum());

    println!("Rank: {} sum = {:?}", rank, count);

    Ok(())
}
