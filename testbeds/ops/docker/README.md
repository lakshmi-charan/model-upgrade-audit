# docker/

Compose files and configuration for the real services (database, issue tracker, mail server, object store or ledger, stream store), the fault proxy and the workflow engine. Data volumes go in `docker/data/` (git-ignored).

- `inventory/` compose file used for the stage-2 smoke tests
