from pathlib import Path

from posttrainx.api.catalog import ProductCatalog


def test_catalog_search():
    root = Path(__file__).resolve().parents[1]
    cat = ProductCatalog(str(root / "data/sample/products.json"))
    results = cat.search("Nike black running shoes")
    assert results
    assert results[0]["brand"] == "Nike"
