"""Synthetic logical software cases; NOT new simulator experiments."""
import unittest
from dataclasses import replace
from claim_contracts import ClaimContract, Verdict, conformance, false_assurance_bounds, sampled_fallback, may_reuse_support


class ReportingLogic(unittest.TestCase):
    def setUp(self):
        self.c = ClaimContract("sampled fallback", 0, 1, "episode 1", ("operator", "neighbor"), "valid sampled state")

    def test_positive_violation_survives_incompleteness(self):
        self.assertEqual(conformance(known_violation=True, coverage_complete=False, all_required_evidence_valid=False), Verdict.REFUTED)

    def test_absence_without_coverage_is_unknown(self):
        self.assertEqual(conformance(known_violation=False, coverage_complete=False, all_required_evidence_valid=True), Verdict.UNRESOLVED)

    def test_zero_acceptance_is_undefined(self):
        self.assertIsNone(false_assurance_bounds(accepted=0, known_violation=0, unresolved=0))

    def test_unresolved_acceptance_widens_bounds(self):
        self.assertEqual(false_assurance_bounds(accepted=5, known_violation=1, unresolved=2), (.2,.6))

    def test_invalid_denominators_rejected(self):
        for kw in [dict(accepted=1,known_violation=1,unresolved=1),dict(accepted=-1,known_violation=0,unresolved=0)]:
            with self.assertRaises(ValueError): false_assurance_bounds(**kw)

    def test_short_hold_does_not_validate_later_horizon(self):
        values=[(0,True),(.5,True),(1,True),(1.5,False),(2,True)]
        self.assertEqual(sampled_fallback(self.c, values, .5), Verdict.SUPPORTED)
        self.assertEqual(sampled_fallback(replace(self.c,end=2),values,.5), Verdict.REFUTED)

    def test_gap_or_unknown_cannot_establish_maintained_response(self):
        for values in [[(0,True),(1,True)],[(0,True),(.5,None),(1,True)],[(.5,True),(1,True)],[]]:
            self.assertEqual(sampled_fallback(self.c, values, .5), Verdict.UNRESOLVED)

    def test_clock_reversal_is_not_supported(self):
        self.assertEqual(sampled_fallback(self.c,[(0,True),(.75,True),(.5,True),(1,True)],1),Verdict.UNRESOLVED)

    def test_claim_scope_cannot_be_promoted(self):
        self.assertFalse(may_reuse_support(self.c,replace(self.c,end=2)))
        self.assertFalse(may_reuse_support(self.c,replace(self.c,denominator="another episode")))
        self.assertFalse(may_reuse_support(self.c,replace(self.c,obligation="moral adequacy")))

    def test_documented_normative_fields_are_not_inferred(self):
        self.assertFalse(self.c.normative_fields_documented)
        self.assertTrue(replace(self.c,normative_justification="proposed rationale",challenge_procedure="proposed review").normative_fields_documented)

    def test_invalid_horizon_rejected(self):
        for end in [-1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError): replace(self.c,end=end)

if __name__ == '__main__': unittest.main()
