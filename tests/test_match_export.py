#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np
import re
import os
from tempfile import TemporaryDirectory

from tests import MOZART_VARIATION_FILES

from partitura.io.exportmatch import matchfile_from_alignment, save_match
from partitura.io.importmatch import load_match
from partitura.io.matchfile_utils import Version
from partitura.io.matchlines_v1 import MatchSection, MatchOmittedSection
from partitura import load_score


class TestExportMatch(unittest.TestCase):
    def test_matchfile_from_alignment(self):
        """
        test `matchfile_from_alignment`
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]

        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        matchfile = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            assume_part_unfolded=True,
        )

        sna = spart.note_array()
        pna = performance.note_array()

        # assert that matchfile contains the same number of notes
        self.assertTrue(len(sna) == len(matchfile.snotes))
        self.assertTrue(len(pna) == len(matchfile.notes))

        snote_ids = [n.Anchor for n in matchfile.snotes]
        pnote_ids = [n.Id for n in matchfile.notes]
        # assert that all snotes in the matchfile are in the note array
        self.assertTrue(all([n.Anchor in sna["id"] for n in matchfile.snotes]))

        # assert that all notes in the score are in the matchfile
        self.assertTrue(all([nid in snote_ids for nid in sna["id"]]))

        # assert that all notes in the matchfile are in the note array
        self.assertTrue(all([n.Id in pna["id"] for n in matchfile.notes]))

        # assert that all notes in the performance are in the matchfile
        self.assertTrue(all(nid in pnote_ids for nid in pna["id"]))

        for ml in matchfile.lines:
            self.assertTrue(isinstance(ml.matchline, str))

    def test_save_match(self):
        """
        Test save_match
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]

        score = load_score(score_fn)
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)
        pna1 = performance.note_array()
        with TemporaryDirectory() as tmpdir:

            out = os.path.join(tmpdir, "test.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=out,
                performer="A Pianist",
                composer="W. A. Mozart",
                piece="mozart_k265_var1",
                score_filename=os.path.basename(score_fn),
                performance_filename=os.path.basename(MOZART_VARIATION_FILES["midi"]),
                assume_unfolded=True,
            )

            perf_from_saved_match, alignment_from_saved_match = load_match(out)

            # Test Metadata
            # Assuming perf_from_saved_match or the parsed mf holds info
            # You may need to load the MatchFile object directly to test info lines easily
            from partitura.io.importmatch import load_matchfile
            mf = load_matchfile(out)
            self.assertEqual(mf.info("composer"), "W. A. Mozart")
            self.assertEqual(mf.info("performer"), "A Pianist")
            self.assertEqual(mf.info("piece"), "mozart_k265_var1")

            # Test Pedal Data
            pedals_original = performance[0].controls
            pedals_saved = perf_from_saved_match[0].controls
            self.assertEqual(len(pedals_original), len(pedals_saved))
            for p1, p2 in zip(pedals_original, pedals_saved):
                self.assertEqual(p1["time"], p2["time"])
                self.assertEqual(p1["value"], p2["value"])

        pna2 = perf_from_saved_match.note_array()

        # Test that performance data is the same
        for field in (
            "onset_sec",
            "duration_sec",
            "onset_tick",
            "duration_tick",
            "pitch",
            "velocity",
        ):
            self.assertTrue(np.allclose(pna2[field], pna1[field]))

        # Test that the alignment info is correct
        for al in alignment_from_saved_match:
            self.assertTrue(al in alignment)

        for al in alignment:
            self.assertTrue(al in alignment_from_saved_match)

    def test_save_match_old_and_new_versions(self):
        """
        Test save_match for version 1.0.0 and a newer version with sections.
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]
        score = load_score(score_fn)
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        section = {
            "id": "section_1",
            "start_in_beats_unfolded": 0.0,
            "end_in_beats_unfolded": 1.0,
            "start_in_beats_original": 0.0,
            "end_in_beats_original": 1.0,
            "start_in_perf_time": 0.0,
            "end_in_perf_time": 1.0,
            "section_attr_list": ["section_attr"],
        }
        omitted_section = {
            "id": "omitted_1",
            "start_in_beats_unfolded": 1.0,
            "end_in_beats_unfolded": 2.0,
            "start_in_beats_original": 1.0,
            "end_in_beats_original": 2.0,
            "section_attr_list": ["omitted_attr"],
        }

        with TemporaryDirectory() as tmpdir:
            old_out = os.path.join(tmpdir, "old_version.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=old_out,
                version=(1, 0, 0),
                sections=[section],
                omitted_sections=[omitted_section],
            )
            with open(old_out, "r", encoding="utf-8") as fh:
                old_content = fh.read()

            self.assertIn("matchFileVersion", old_content)
            self.assertNotIn("section_1", old_content)
            self.assertNotIn("omitted_1", old_content)

            new_out = os.path.join(tmpdir, "new_version.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=new_out,
                version=(1, 1, 0),
                sections=[section],
                omitted_sections=[omitted_section],
            )
            with open(new_out, "r", encoding="utf-8") as fh:
                new_content = fh.read()

            self.assertIn("section_1", new_content)
            self.assertIn("omitted_1", new_content)

    def test_matchfile_from_alignment_version_specific_lines(self):
        """
        Test that version 1.0.0 ignores section lines, while newer versions emit them.
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]
        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        section = {
            "id": "section_1",
            "start_in_beats_unfolded": 0.0,
            "end_in_beats_unfolded": 1.0,
            "start_in_beats_original": 0.0,
            "end_in_beats_original": 1.0,
            "start_in_perf_time": 0.0,
            "end_in_perf_time": 1.0,
            "section_attr_list": ["section_attr"],
        }
        omitted_section = {
            "id": "omitted_1",
            "start_in_beats_unfolded": 1.0,
            "end_in_beats_unfolded": 2.0,
            "start_in_beats_original": 1.0,
            "end_in_beats_original": 2.0,
            "section_attr_list": ["omitted_attr"],
        }

        mf_old = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            version=Version(1, 0, 0),
            sections=[section],
            omitted_sections=[omitted_section],
        )
        self.assertFalse(
            any(
                isinstance(line, (MatchSection, MatchOmittedSection))
                for line in mf_old.lines
            )
        )

        mf_new = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            version=Version(1, 1, 0),
            sections=[section],
            omitted_sections=[omitted_section],
        )
        self.assertTrue(any(isinstance(line, MatchSection) for line in mf_new.lines))
        self.assertTrue(
            any(isinstance(line, MatchOmittedSection) for line in mf_new.lines)
        )
    
    def test_virtual_notes_export_version_1_1_0(self):
        """
        Test the export of virtual notes introduced in version 1.1.0.
        Virtual notes occur implicitly when the same score_id or performance_id 
        is aligned multiple times.
        """
        from partitura.io.matchlines_v1 import (
            MatchSnoteNote,
            MatchVirtualSnoteNote,
            MatchSnoteVirtualNote,
            MatchVirtualSnoteVirtualNote
        )
        from partitura.io.importmatch import load_matchfile

        score_fn = MOZART_VARIATION_FILES["musicxml"]
        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, original_alignment = load_match(match_fn)
        
        # Extract two distinct valid matches to get authentic IDs
        matches = [a for a in original_alignment if a["label"] == "match"]
        m1, m2 = matches[0], matches[1]
        
        s1, p1 = m1["score_id"], m1["performance_id"]
        s2, p2 = m2["score_id"], m2["performance_id"]
        
        # Create a synthetic alignment to force the exporter into all 4 states:
        synthetic_alignment = [
            # 1. Standard (neither seen before) -> MatchSnoteNote
            {"label": "match", "score_id": s1, "performance_id": p1}, 
            
            # 2. Score ID seen, Perf ID new -> MatchVirtualSnoteNote
            {"label": "match", "score_id": s1, "performance_id": p2}, 
            
            # 3. Score ID new, Perf ID seen -> MatchSnoteVirtualNote
            {"label": "match", "score_id": s2, "performance_id": p1}, 
            
            # 4. Both IDs seen before -> MatchVirtualSnoteVirtualNote
            {"label": "match", "score_id": s1, "performance_id": p1}, 
        ]
        
        # Export using 1.1.0 logic
        mf = matchfile_from_alignment(
            alignment=synthetic_alignment,
            ppart=performance[0],
            spart=spart,
            version=Version(1, 1, 0),
            assume_part_unfolded=True
        )
        
        # Extract the classes of the generated lines to verify the routing logic worked
        line_types = [type(line) for line in mf.lines]
        
        self.assertIn(MatchSnoteNote, line_types, "Standard MatchSnoteNote missing")
        self.assertIn(MatchVirtualSnoteNote, line_types, "MatchVirtualSnoteNote missing")
        self.assertIn(MatchSnoteVirtualNote, line_types, "MatchSnoteVirtualNote missing")
        self.assertIn(MatchVirtualSnoteVirtualNote, line_types, "MatchVirtualSnoteVirtualNote missing")
        
        # Perform a round-trip IO check to ensure the parsers successfully load these lines back
        with TemporaryDirectory() as tmpdir:
            out_fn = os.path.join(tmpdir, "virtual_test.match")
            mf.write(out_fn)
            
            parsed_mf = load_matchfile(out_fn)
            parsed_types = [type(line) for line in parsed_mf.lines]
            
            self.assertIn(MatchVirtualSnoteNote, parsed_types, "Failed to parse MatchVirtualSnoteNote")
            self.assertIn(MatchSnoteVirtualNote, parsed_types, "Failed to parse MatchSnoteVirtualNote")
            self.assertIn(MatchVirtualSnoteVirtualNote, parsed_types, "Failed to parse MatchVirtualSnoteVirtualNote")
