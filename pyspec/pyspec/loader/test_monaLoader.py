from pyspec.loader.mona import MoNALoader, MoNAQueryGenerator


def test_load_record():
    mona = MoNALoader()
    data = mona.load_spectrum('AU101801')

    assert(data.id == 'AU101801')
    assert(len(data.properties) == 4)

def test_query_by_splash():
    mona = MoNALoader()
    query = MoNAQueryGenerator().query_by_splash('splash10-0a4i-3910000000-a82697559a690c6121ea')

    data = mona.query(query)

    assert(len(data) >= 1)
    assert(any(s.id == 'AU101801' for s in data))

def test_query_by_splash():
    mona = MoNALoader()
    query = MoNAQueryGenerator().query_by_splash('splash10-0a4i-3910000000-a82697559a690c6121ea')

    data = mona.query(query)

    assert(len(data) >= 1)
    assert(any(s.id == 'AU101801' for s in data))