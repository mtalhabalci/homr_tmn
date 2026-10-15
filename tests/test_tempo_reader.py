from homr.tempo_reader import too_wide


def test_a_line_of_words_and_notes_is_too_wide_for_a_tempo_mark() -> None:
    # Aydemir p.82: '1.Hane' run into the notes above the first staff (staff 69 px high)
    assert too_wide((442, 578, 1501, 641), 69)


def test_a_printed_tempo_mark_is_not_too_wide() -> None:
    # an eighth = 112 above a 69 px staff: about 3.5 staff heights wide, two thirds of one high
    assert not too_wide((500, 560, 740, 605), 69)
