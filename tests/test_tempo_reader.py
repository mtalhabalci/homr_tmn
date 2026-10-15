from homr.tempo_reader import numbers, playable, too_wide


def test_a_line_of_words_and_notes_is_too_wide_for_a_tempo_mark() -> None:
    # Aydemir p.82: '1.Hane' run into the notes above the first staff (staff 69 px high)
    assert too_wide((442, 578, 1501, 641), 69)


def test_a_printed_tempo_mark_is_not_too_wide() -> None:
    # an eighth = 112 above a 69 px staff: about 3.5 staff heights wide, two thirds of one high
    assert not too_wide((500, 560, 740, 605), 69)


def test_a_quarter_at_728_is_no_tempo_but_an_eighth_at_440_is() -> None:
    assert not playable("U4", False, 728)
    assert playable("U8", False, 440)
    assert playable("U4", True, 60)
    assert not playable("U16", False, 60)


def test_a_range_is_read_as_its_two_ends_and_a_bare_number_as_one() -> None:
    assert numbers(["U4", "7", "2", "-", "8", "4"]) == (72, 84)
    assert numbers(["U8", "1", "6", "0"]) == (160, None)
    assert numbers(["U4", "8", "4", "-", "7", "2"]) is None
    assert numbers(["U4", "7", "2", "-"]) is None
