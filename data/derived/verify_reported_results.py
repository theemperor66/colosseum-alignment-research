"""Verify published arithmetic from saved endpoint labels. Does not recreate labels from raw simulator records."""
from pathlib import Path
import json,math,hashlib
from collections import Counter,defaultdict
D=Path(__file__).resolve().parent
load=lambda n:json.loads((D/n).read_text())
e=load('endpoint-rows.json');s=load('scientific-summary.json');old=load('original-endpoint-rows.json');olds=load('original-scientific-summary.json');rows=e['results']
assert len(rows)==792 and e['observed_attempts']==792 and len({r['episode_id'] for r in rows})==792
assert Counter(r['attempt']['status'] for r in rows)=={'completed':792}
assert sum(bool(r['completion_pass']) for r in rows)==35
byid={r['episode_id']:r for r in rows}
assert sum(r['attempt']['status']=='completed' for r in old['results'])==757
for r in old['results']:
 if r['attempt']['status']=='completed':assert r==byid[r['episode_id']]
def contrast(rows,endpoint,left,right,subset):
 groups=defaultdict(dict)
 for r in rows:
  if r[subset]:groups[(r['protocol_hash'],r['scenario_id'])][r['arm_id']]=r[endpoint]
 pairs=[(g[left][0]-g[right][1],g[left][1]-g[right][0]) for g in groups.values()];n=len(pairs);b=[sum(a for a,z in pairs)/n,sum(z for a,z in pairs)/n];h=math.sqrt(2*math.log(2/.05)/n)
 return {'n':n,'bounds':b,'interval':[max(-1,b[0]-h),min(1,b[1]+h)]}
for rr,summary,endpoints in [(rows,s,e),(old['results'],olds,old)]:
 for k,key in [('physical','physical_violation_bounds'),('procedural','procedural_violation_bounds'),('safe_useful','safe_useful_mission_bounds')]:
  v=contrast(rr,key,'A1_policy_only','A2_assumption_aware','primary');assert v['n']==180;ref=summary['planned_primary_comparisons'][k];assert v['bounds']==ref['identification_bounds'] and v['interval']==ref['marginal_95_hoeffding_interval']
 for c in endpoints['aggregate']['focused_secondary']:
  v=contrast(rr,c['endpoint'],c['left'],c['right'],'focused');assert v['n']==48 and v['bounds']==c['identification_bounds'] and v['interval']==c['marginal_95_hoeffding_interval']
for arm,summary in s['arms'].items():
 rr=[r for r in rows if r['arm_id']==arm];assert len(rr)==summary['observed']==summary['planned']
 for short,key in [('physical','physical_violation_bounds'),('procedural','procedural_violation_bounds'),('safe_useful','safe_useful_mission_bounds')]:assert [sum(r[key][i] for r in rr) for i in (0,1)]==summary[short]
 assert sum(r['native_outcome']['accepted_by_monitor'] is True for r in rr)==summary['native_assurance']['accepted']
for name,binding in load('derivation-provenance.json').items():
 if isinstance(binding,dict):assert hashlib.sha256((D/name).read_bytes()).hexdigest()==binding['derived_sha256'],name
account=load('completion-accounting.json');assert account['completed_design_slots']==792 and account['original_completed_preserved']==757 and account['selected_followup_slots']==35
assert account['total_recorded_attempts_including_original']==789+account['total_new_attempt_records']
repro=load('separate-host-followup-verification.json');assert sum(g['complete_records_audited'] for b in repro['batch_receipts'] for g in b['reproduction']['groups'])==35
assert repro['completed_combination']['verification']['all_secondary_aggregates_equal']
result={'all_checks_passed':True,'planned':792,'complete':792,'original_complete_preserved':757,'first_complete_followups':35,'full_attempt_history':account['total_recorded_attempts_including_original'],'primary':contrast(rows,'physical_violation_bounds','A1_policy_only','A2_assumption_aware','primary'),'primary_group_table_contrasts_verified':3,'focused_contrasts_verified':6,'original_sensitivity_contrasts_verified':9,'arm_aggregate_tables_verified':True,'derivative_hashes_verified':True,'scope':'Arithmetic and accounting from saved labels, including the unchanged original sensitivity analysis; no raw-data rescoring or new experiments.'}
print(json.dumps(result,indent=2))
