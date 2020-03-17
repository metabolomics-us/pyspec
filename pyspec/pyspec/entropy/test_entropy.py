import pytest
from pandas import read_sql_query, DataFrame
from tabulate import tabulate

from pyspec.entropy.entropy import Entropy
from pyspec.loader import Spectra
from pyspec.machine.persistence.model import db


def test_compute():
    """
    tests if the compute function will be executed right and returns the correct results
    """

    spectra = Spectra(
        ms_level=1,
        spectra="100:0.5 101:0.5"
    )

    entropy = Entropy()
    score = entropy.compute(spectra)

    assert score is not None
    assert score == (1.0, 2)

    print(score)


@pytest.mark.parametrize("cutoff", [0, 3, 5])
def test_compute_from_ml_database(cutoff):
    """
    computes a large amount of entropies from the database
    and ensure none of them fails
    """

    entropy = Entropy()

    result = read_sql_query(
        "select * from mzmlmsmsspectrarecord m , mzmzmsmsspectraclassificationrecord m2, mzmlsamplerecord m3  where m.id = m2.spectra_id and m.sample_id = m3.id and precursor_intensity > 0 and m2.category = 'origin' and m2.value = 'lc-binbase' order by m.sample_id asc",
        db.connection())

    data = []
    for index, row in result.iterrows():
        spectra = Spectra(
            ms_level=row['level'],
            spectra=row['msms'],
            properties={
                'pre_cursor': row['precursor'],
                'pre_cursor_intensity': row['precursor_intensity']
            }
        )

        e = entropy.compute(spectra, min_intensity=cutoff)
        if e[1] == 0:
            pass
        data.append(
            {
                "entropy": e[0],
                "precursor": row['precursor'],
                "intensity": row['precursor_intensity'],
                "ioncount": e[1]
            }
        )

    frame = DataFrame(data)

    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    ax = sns.scatterplot(x="intensity", y="entropy", hue="ioncount", data=frame)

    plt.show()
