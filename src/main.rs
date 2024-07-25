use distributed_louvain::{DistributedGraph, Edge};

use anyhow;
use mpi::traits::Communicator;

fn main() -> anyhow::Result<()> {
    // let global_edges = [
    //     // rank 0
    //     Edge(1, 2, 1.0),
    //     Edge(1, 4, 1.0),
    //     Edge(1, 7, 1.0),
    //     Edge(2, 0, 1.0),
    //     Edge(2, 4, 1.0),
    //     Edge(2, 5, 1.0),
    //     Edge(2, 6, 1.0),
    //     // rank 1
    //     Edge(3, 0, 1.0),
    //     Edge(3, 7, 1.0),
    //     Edge(4, 0, 1.0),
    //     Edge(4, 10, 1.0),
    //     Edge(5, 0, 1.0),
    //     Edge(5, 7, 1.0),
    //     Edge(5, 11, 1.0),
    //     // rank 2
    //     Edge(6, 7, 1.0),
    //     Edge(6, 11, 1.0),
    //     Edge(8, 9, 1.0),
    //     Edge(8, 10, 1.0),
    //     Edge(8, 11, 1.0),
    //     Edge(8, 14, 1.0),
    //     Edge(8, 15, 1.0),
    //     // rank 3
    //     Edge(9, 12, 1.0),
    //     Edge(9, 14, 1.0),
    //     Edge(10, 11, 1.0),
    //     Edge(10, 12, 1.0),
    //     Edge(10, 13, 1.0),
    //     Edge(10, 14, 1.0),
    //     Edge(11, 13, 1.0),
    // ];

    // let (_, world) = distributed_louvain::init()?;
    // let rank = world.rank();

    // let edges = match rank {
    //     0 => &global_edges[0..7],
    //     1 => &global_edges[7..14],
    //     2 => &global_edges[14..21],
    //     3 => &global_edges[21..28],
    //     _ => panic!("Too many processes"),
    // };

    // let dg = DistributedGraph::from_distributed(&edges, &world)?;

    // println!("{:?}", dg);

    Ok(())
}
