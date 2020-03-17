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
    assert score == 1.0

    print(score)


def test_compute_from_ml_database():
    """
    computes a large amount of entropies from the database
    and ensure none of them fails
    """

    entropy = Entropy()

    result = read_sql_query(
        "select * from  mzmlmsmsspectrarecord where precursor_intensity > 0 limit 1000",
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

        data.append({
            "entropy": entropy.compute(spectra),
            "precursor" : row['precursor'],
            "intensity" : row['precursor_intensity']
        }
        )

    frame = DataFrame(data)

    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    ax = sns.scatterplot(x="intensity", y="entropy", data=frame)

    plt.show()
