use distributed_louvain::DistributedGraph;

use petgraph::csr;

use anyhow;

fn main() -> anyhow::Result<()> {
    let edges = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
        (1, 1, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (1, 4, 1.0),
    ];

    let dg = DistributedGraph::from_distributed(&edges)?;

    println!("{:?}", g);

    Ok(())
}
