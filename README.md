# Audio identification by fingerprinting
Given a short audio fragment – the query – and a database
of complete recordings, the audio identification task consists
of retrieving the original recording the query is reproducing.
This matching process is conducted by comparing
specifically designed audio features called fingerprints.

An audio identification algorithm was implemented
based on the one described by Wang et al. in [1] from
the company Shazam 1 , as well as the remarks made by
Müller on [2], and adapted to return the best three matches
from the relevant database.

The system is designed to operate on a subset of the
GTZAN dataset, comprising 100 pop recordings and 100
classical recordings. 108 degraded snippets from the classical
recordings and 105 degraded snippets from the pop
recordings were used as query examples. The snippets aim
to simulate the real-world conditions in which the system
would be used, containing distorted or filtered audio at a
low volume, combined with everyday sounds.

## References
[1] A. Wang et al., “An industrial strength audio search algorithm.”
in Ismir, vol. 2003. Washington, DC, 2003,
pp. 7–13.
[2] M. Müller, Fundamentals of music processing: Audio,
analysis, algorithms, applications. Springer, 2015,
vol. 5.
