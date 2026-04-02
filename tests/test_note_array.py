#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the test cases for testing the note_array attribute of
the Part class.
"""

import unittest

import partitura.score as score
from partitura import load_musicxml, load_kern, load_score
from partitura.utils.music import note_array_from_part, ensure_notearray, expand_grace_notes_from_local_grace_order, remove_double_notes_from_score
from partitura.musicanalysis import note_array_to_score
import numpy as np

from tests import NOTE_ARRAY_TESTFILES, KERN_TESTFILES, METRICAL_POSITION_TESTFILES, GRACE_NOTE_DOC_ORDER_TESTFILES, DOUBLE_NOTE_TESTFILE


class TestNoteArray(unittest.TestCase):
    """
    Test the note_array attribute of the Part class
    """

    def test_notearray_1(self):
        part = score.Part("P0", "My Part")

        part.set_quarter_duration(0, 10)
        part.add(score.TimeSignature(3, 4), start=0)
        part.add(score.Note(id="n0", step="A", octave=4), start=0, end=10)

        note_array = part.note_array()
        self.assertTrue(len(note_array) == 1)

    def test_notearray_beats(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])[0]
        note_array = score.note_array()
        expected_onset_beats = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 30, 32]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))

    def test_notearray_musical_beats1(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        score[0].use_musical_beat()
        note_array = note_array_from_part(score[0])
        expected_onset_beats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))
        self.assertTrue(
            np.array_equal(score[0].note_array()["onset_beat"], expected_onset_beats)
        )

    def test_use_musical_beats1(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        score[0].use_musical_beat({"6/8": 3, "2/4": 1})
        note_array = note_array_from_part(score[0])
        expected_onset_beats = [0, 1.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.5, 12.5]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))
        self.assertTrue(
            np.array_equal(score[0].note_array(), note_array_from_part(score[0]))
        )

    def test_use_musical_beats2(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        # set musical beats
        score[0].use_musical_beat()
        note_array = score[0].note_array()
        self.assertTrue(score[0]._use_musical_beat == True)
        # unset musical beats
        score[0].use_notated_beat()
        self.assertTrue(score[0]._use_musical_beat == False)
        self.assertFalse(np.array_equal(score[0].note_array(), note_array))

    def test_note_array_to_score_multiple_ts(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = score.note_array(include_time_signature=True)
        new_score = note_array_to_score(note_array)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

    def test_note_array_to_score_anacrusis(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        new_score = note_array_to_score(note_array)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

    def test_note_array_to_score_only_divs(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        for div in [1,2,4,8]:
            new_score = note_array_to_score(note_array[["onset_div", "duration_div", "pitch"]],divs = div)
            new_note_array = new_score.note_array(include_time_signature=True)
            self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_div"]/div))
            self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_div"]/div))

    def test_note_array_to_score_only_div_time_sig(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        for time_sig in [4,8,9]:
            new_score = note_array_to_score(note_array[["onset_div", "duration_div", "pitch"]],divs = 1, time_sigs = [(2,4, time_sig)])
            new_note_array = new_score.note_array(include_time_signature=True)
            self.assertTrue(np.all(new_note_array["ts_beat_type"] == np.ones_like(note_array["onset_div"])*time_sig))

    def test_note_array_to_score_only_beats(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        # without time signature -> barebones score without measures/anacrusis
        new_score = note_array_to_score(note_array[["onset_beat", "duration_beat", "pitch"]])
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]+1.0))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

        # with time signature -> measures/anacrusis kept, but div/beat ratio might change
        new_score = note_array_to_score(note_array[["onset_beat", "duration_beat", "pitch"]], time_sigs= [(1,3,8)] )
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]*2))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]*2))

        # with default time signature -> measures/anacrusis kept, 4/4 corrsponds to original
        new_score = note_array_to_score(note_array[["onset_beat", "duration_beat", "pitch"]], estimate_time = True)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

    def test_note_array_to_score_beats_and_div(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        # without time signature -> barebones score without measures/anacrusis
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch"]])
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]+1.0))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

        # with time signature -> measures/anacrusis kept, but div/beat ratio might change
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch"]], time_sigs= [(1,3,8)] )
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]*2))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]*2))

        # with default time signature -> measures/anacrusis kept, 4/4 corrsponds to original
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch"]], estimate_time = True)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

        # with original time signature -> measures/anacrusis kept, other arguments overridden
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch", "ts_beats", "ts_beat_type"]] , time_sigs= [(1,3,8)], estimate_time = True)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))


    def test_notearray_ts_beats(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])[0]
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_beats = [6, 6, 9, 9, 9, 12, 12, 12, 12, 2, 2, 2, 2]
        self.assertTrue(np.array_equal(note_array["ts_beats"], expected_beats))

        # now using musical beats
        part.use_musical_beat()
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_musical_beats = [2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2]
        self.assertTrue(
            np.array_equal(note_array["ts_mus_beats"], expected_musical_beats)
        )

    def test_ensure_na_different_divs(self):
        # check if divs are correctly rescaled when producing a note array from
        # parts with different divs values
        parts = load_kern(KERN_TESTFILES[7]).parts
        merged_note_array = ensure_notearray(parts)
        for note in merged_note_array[-4:]:
            self.assertTrue(note["onset_div"] == 2208)
            self.assertTrue(note["duration_div"] == 96)
            self.assertTrue(note["divs_pq"] == 96)

    def test_score_notearray_method(self):
        """
        Test that note array generated from the Score class method
        include all relevant information.
        """

        for fn in NOTE_ARRAY_TESTFILES:

            scr = load_score(fn)

            na = scr.note_array(
                include_pitch_spelling=True,
                include_key_signature=True,
                include_time_signature=True,
                include_grace_notes=True,
                include_metrical_position=True,
                include_staff=True,
                include_divs_per_quarter=True,
            )

            expected_field_names = [
                "onset_beat",
                "duration_beat",
                "onset_quarter",
                "duration_quarter",
                "onset_div",
                "duration_div",
                "pitch",
                "voice",
                "id",
                "step",
                "alter",
                "octave",
                "is_grace",
                "grace_type",
                "ks_fifths",
                "ks_mode",
                "ts_beats",
                "ts_beat_type",
                "ts_mus_beats",
                "is_downbeat",
                "rel_onset_div",
                "tot_measure_div",
                "staff",
                "divs_pq",
            ]

            for field_name in expected_field_names:
                # check that the note array contain the relevant
                # field.
                self.assertTrue(field_name in na.dtype.names)

    def test_gracenote_doc_order(self):
        """
        Test that the document order of grace notes is correctly reflected in the note array.
        """
        for fn in GRACE_NOTE_DOC_ORDER_TESTFILES:
            score_part = load_musicxml(fn)[0]
            sna = score_part.note_array(include_grace_notes=True)

            expected_field_names = [
                "is_grace",
                "grace_type",
                "steal_proportion",
                "local_grace_order",
            ]
            
            for field_name in expected_field_names:
                self.assertTrue(field_name in sna.dtype.names)
            
            grace_notes_array = sna[sna["is_grace"] == 1]
            local_grace_order_values = grace_notes_array["local_grace_order"]

            if 'non_chord' in fn:
                # check that all the local_grace_order_values are unique
                self.assertTrue(len(local_grace_order_values)==len(set(local_grace_order_values)))
            
            elif 'mixed_chord' in fn:
                # check that the set of local_grace_order_values is equal to 4
                self.assertTrue(len(local_grace_order_values) == 7)
                self.assertTrue(set(local_grace_order_values) == set([0,1,2,3]))
            
            else:

                # check that all the local_grace_order values are the same
                self.assertTrue(np.all(local_grace_order_values == local_grace_order_values[0]))

    def test_expand_grace_notes_from_local_grace_order(self):
        """
        Test that the expand_grace_notes_from_local_grace_order function correctly expands the grace notes in the note array based on their local grace order.
        """
        
        for fn in GRACE_NOTE_DOC_ORDER_TESTFILES:
            score_part = load_musicxml(fn)[0]
            expanded_note_array = expand_grace_notes_from_local_grace_order(score_part, grace_offset_quarter=0.25)

            if 'non_chord' in fn:
                # check that the grace notes are expanded by 0.25 quarter notes before the main note
                main_note_onset = expanded_note_array[expanded_note_array["is_grace"] == 0]["onset_beat"][0]
                grace_note_onsets = expanded_note_array[expanded_note_array["is_grace"] == 1]["onset_beat"]
                grace_note_durations = expanded_note_array[expanded_note_array["is_grace"] == 1]["duration_beat"]
                grace_notes_length = len(grace_note_onsets)

                self.assertTrue(grace_note_onsets[0] == main_note_onset - 0.25)
                self.assertTrue(np.all(grace_note_durations == 0.25/grace_notes_length))

                # convert all the grace note onsets to .2f values
                grace_note_onsets_rounded = np.round(grace_note_onsets, 2)
                true_onsets = []
                for i in range(grace_notes_length):
                    true_onsets.append(np.round(main_note_onset - 0.25 + i*(0.25/grace_notes_length), 2))

                # check that all the values of grace_note_onsets_rounded are equal to the corresponding values in true_onsets
                self.assertTrue(np.allclose(grace_note_onsets_rounded, true_onsets))

            elif 'mixed_chord' in fn:
                main_note_onset = expanded_note_array[expanded_note_array["is_grace"] == 0]["onset_beat"][0]
                grace_note_onsets = expanded_note_array[expanded_note_array["is_grace"] == 1]["onset_beat"]
                unique_grace_note_onsets = set(grace_note_onsets)
                
                # check that the unique grace onsets are not equal to all the grace note onsets, as there is a chordal grace note
                self.assertTrue(len(unique_grace_note_onsets) == 4)
                self.assertTrue(len(grace_note_onsets) == 7)
                
                grace_note_durations = expanded_note_array[expanded_note_array["is_grace"] == 1]["duration_beat"]
                grace_notes_length = len(grace_note_onsets)
                unique_grace_notes_length = len(unique_grace_note_onsets)

                # check that the durations of the grace notes with the same onset are equal, 
                # and that they are equal to 0.25 divided by the number of unique grace note onsets
                self.assertTrue(np.all(grace_note_durations == 0.25/unique_grace_notes_length))

                # validate first grace note onset
                self.assertTrue(grace_note_onsets[0] == main_note_onset - 0.25)

                true_onsets = np.array([-0.25, -0.1875, -0.125, -0.0625, -0.0625, -0.0625, -0.0625])

                self.assertTrue(np.allclose(grace_note_onsets, true_onsets))


    def test_remove_double_notes_from_score(self):
        """
        Test that the remove_double_notes_from_score function correctly removes double notes from the score.
        """
        score_part = load_musicxml(DOUBLE_NOTE_TESTFILE)[0]
        sna = score_part.note_array()

        for i in range(2):
            if i == 0:
                sna_no_double = remove_double_notes_from_score(score_part)
            else:
                sna_no_double = remove_double_notes_from_score(score_part, choose_longer_note=True)
            
            # check that the number of notes in sna_no_double is less than the number of notes in sna
            self.assertTrue(len(sna) - len(sna_no_double) == 3)

            sna_ids = set(sna["id"])
            sna_no_double_ids = set(sna_no_double["id"])
            sna_double_only_ids = sna_ids - sna_no_double_ids
            sna_double_only = sna[np.isin(sna["id"], list(sna_double_only_ids))]
            
            # test the 'choose_longer_note' functionality
            for note in sna_double_only:
                onset = note["onset_beat"]
                pitch = note["pitch"]
                duration = note["duration_beat"]

                corresponding_notes = sna[(sna["onset_beat"] == onset) & (sna["pitch"] == pitch) & (sna["id"] != note["id"])]

                self.assertTrue(len(corresponding_notes) == 1)
                if i == 0:    
                    self.assertTrue(duration > corresponding_notes["duration_beat"][0])
                else:
                    self.assertTrue(duration < corresponding_notes["duration_beat"][0])
    






if __name__ == "__main__":
    unittest.main()
