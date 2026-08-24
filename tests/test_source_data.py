from ai4us.source_data import validate_source_data


def test_all_quantitative_sheets_match_frozen_csv_aliases() -> None:
    report = validate_source_data()
    assert report["status"] == "PASS"
    assert report["sheet_count"] == 16
    assert report["cell_matrix_equal_count"] == 16
    assert report["formula_cell_count"] == 0
