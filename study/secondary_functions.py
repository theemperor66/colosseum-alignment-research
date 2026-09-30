from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean,median
from datetime import datetime,timezone
import hashlib,json,math
R=Path('/work'); D=Path('/receipt')
LOCAL=R/'output/fresh-worker-53039233-20260927/pool-final588-review/closed-588-endpoints-v1.json'
LINUX=R/'output/closure-receipts-20260928/cpu588-attempt02/rescored-588.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distribution(values):
 values=sorted(float(v) for v in values if v is not None)
 assert all(math.isfinite(x) for x in values)
 return {'n':len(values),'min':min(values) if values else None,'median':median(values) if values else None,'mean':mean(values) if values else None,'max':max(values) if values else None}
def count(values):return dict(sorted(Counter(values).items()))
def summarize(rows):
 obs=[r for r in rows if r['status']=='scored'];out=[r['original_independent_outcome'] for r in obs];measured=[o for o in out if o.get('measurements') is not None]
 violation=[o for o in out if o['any_violation']];monitored=[o for o in violation if o['accepted_by_monitor'] is not None];detect=[o for o in monitored if o['missed_detection'] is not None]
 exp=[o['measurements']['envelope_exposure'] for o in measured if o['measurements'].get('envelope_exposure') is not None]
 for e in exp:
  assert -1e-7<=e['observed_outside_duration_s']<=e['covered_duration_s']+1e-7
  assert abs(e['covered_duration_s']+e['unobserved_duration_s']-e['eligible_duration_s'])<1e-5
 latency=defaultdict(list); rec=[]
 for o in measured:
  for l in o['measurements']['latencies']:
   if l['status']=='observed':assert l['latency_s'] is not None and l['latency_s']>=0
   else:assert l['latency_s'] is None
   latency[l['metric']].append((o['episode_id'],l))
  rec.extend((o['episode_id'],f) for f in o['measurements']['fault_recoveries'])
 lat={}
 for metric,records in sorted(latency.items()):
  epmeans=defaultdict(list)
  for ep,v in records:
   if v['status']=='observed':epmeans[ep].append(v['latency_s'])
  lat[metric]={'windows':len(records),'episodes_with_windows':len({e for e,l in records}),'status_counts':count(l['status'] for e,l in records),'source_counts':count(l['source'] for e,l in records),'observed_window_latency_s':distribution(l['latency_s'] for e,l in records if l['status']=='observed'),'episode_mean_observed_latency_s':distribution(mean(v) for v in epmeans.values()),'nonobserved_reasons':count(l['reason'] for e,l in records if l['status']!='observed')}
 statuses=sorted({r['status'] for e,r in rec})
 result={'planned_slots':len(rows),'recovered_attempts':len(obs),'unrecovered_slots':len(rows)-len(obs),'technical_status':count(r['attempt']['status'] for r in obs),'evaluator_completeness':count(o['completeness'] for o in out),'measurement_records':len(measured),'records_without_measurements':len(out)-len(measured),
  'interventions':{'total':sum(o['interventions'] for o in out),'episodes_with_any':sum(o['interventions']>0 for o in out),'per_episode':distribution(o['interventions'] for o in out),'kind_entries':count(k for o in out for k in o['diagnostics']['intervention_kinds']),'suspended_episodes':sum(o['suspended'] for o in out),'abandoned_episodes':sum(o['abandoned'] for o in out)},
  'first_violation_detection':{'all_independent_violation_episodes':len(violation),'monitored_violation_episodes':len(monitored),'ascertainable_nearest_report_status':len(detect),'undefined_nearest_report_status':len(monitored)-len(detect),'missed_at_nearest_report':sum(o['missed_detection'] for o in detect),'recorded_later_or_nearest_violation':sum(o['detection_delay_s'] is not None for o in monitored),'no_recorded_later_violation':sum(o['detection_delay_s'] is None for o in monitored),'nearest_miss_with_later_detection':sum(o['missed_detection'] is True and o['detection_delay_s'] is not None for o in detect),'detected_delay_s':distribution(o['detection_delay_s'] for o in monitored),'nearest_report_verdicts':count(str(o['monitor_verdict_at_violation']) for o in monitored)},
  'geofence_exposure':{'measurement_episodes':len(exp),'full_horizon_estimate_available':sum(e['full_horizon_estimate_s'] is not None for e in exp),'episodes_with_positive_sample_held_outside_duration':sum(e['observed_outside_duration_s']>0 for e in exp),'totals':{k:sum(e[k] for e in exp) for k in ['eligible_duration_s','covered_duration_s','unobserved_duration_s','observed_outside_duration_s']},'episode_distributions':{k:distribution(e[k] for e in exp) for k in ['eligible_duration_s','covered_duration_s','unobserved_duration_s','observed_outside_duration_s']},'sum_sampled_excursions':sum(e['sampled_excursions'] for e in exp)},'latencies':lat,
  'fault_recovery':{'windows':len(rec),'episodes_with_fault_windows':len({e for e,f in rec}),'status_counts':count(f['status'] for e,f in rec),'episodes_with_status':{s:len({e for e,f in rec if f['status']==s}) for s in statuses},'windows_with_overlap':sum(f['overlapping_faults']>0 for e,f in rec),'fault_type_counts':count(f['fault_type'] for e,f in rec),'channel_counts':count(f['channel'] for e,f in rec),'source_counts':count(f['source'] for e,f in rec),'observed_breach_windows':sum(f['observed_breach'] for e,f in rec),'recovered_after_breach_latency_s':distribution(f['recovery_latency_s'] for e,f in rec if f['status']=='recovered'),'maintained_safe_post_fault_hold_start_delay_s':distribution(f['recovery_latency_s'] for e,f in rec if f['status']=='maintained_safe'),'no_recovery_rows_reason':count(str(o['measurements']['recovery_unavailable_reason']) for o in measured if not o['measurements']['fault_recoveries'])}}
 return result
def aggregate(data):
 rows=data['results'];arms=sorted({r['arm_id'] for r in rows});cells=sorted({r['cell_id'] for r in rows})
 return {'all':summarize(rows),'per_arm':{a:summarize([r for r in rows if r['arm_id']==a]) for a in arms},'per_cell_arm':{c:{a:summarize([r for r in rows if r['cell_id']==c and r['arm_id']==a]) for a in arms if any(r['cell_id']==c and r['arm_id']==a for r in rows)} for c in cells}}
