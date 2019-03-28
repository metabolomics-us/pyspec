#PySpec

this is a simple tool collecting metadata about LCMC MSMS Spectra and computing some statistics. It's supposed
to run in a docker-swarm cluster and utilizes grafana for visualizations.

#Testing

this utilizes pytest and can be run with

```.env
 pytest . -ss
```

from the root folder. This assumes that you have access to the testfiles, which are partially located on private servers.
