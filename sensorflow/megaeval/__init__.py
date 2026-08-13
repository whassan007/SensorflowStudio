"""megaeval — aggregate-first, risk-first evaluation layer for mega-scale populations.

Mental model (maps 1:1 to modules):

    Dataset (immutable population)   -> population.py   (partitioned npz storage)
    Evaluation Run (first-class job) -> runs.py         (async workers -> partial stats -> reduce)
    Population / Cohort              -> cube.py         (metric cube + query router + cache)
    Container                        -> runs.py         (container-level aggregate table)
    Annotation (forensic drill-down) -> runs.py         (per-object partitions, loaded on demand)

Single-node stand-ins for distributed infra (see ARCHITECTURE.md):
    npz partitions      ~ Parquet files in an Iceberg table
    worker thread pool  ~ Spark executors / Flink task slots
    partial-stats+reduce~ map-side combine + shuffle reduce
    QueryRouter         ~ OLAP engine (Druid/Pinot) in front of the lakehouse
"""
