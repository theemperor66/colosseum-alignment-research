"""Read-only, standard-library audit of retained guard-event/command relationships."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, hashlib, json, zipfile


def digest(b):
    return hashlib.sha256(b).hexdigest()


def summarize(rows):
    missing = [r for r in rows if not r['exact_linked_dispatches']]
    return {
        'requests': len(rows),
        'episodes': len({r['episode_id'] for r in rows}),
        'exact_linked_requests': len(rows) - len(missing),
        'unlinked_requests': len(missing),
        'unlinked_categories': dict(sorted(Counter(r['category'] for r in missing).items())),
        'unlinked_request_kinds': dict(sorted(Counter(r['request_kind'] for r in missing).items())),
        'unlinked_first_of_kind': dict(sorted(Counter(str(r['first_of_kind']) for r in missing).items())),
        'unlinked_by_request_and_command': dict(sorted(Counter(
            '|'.join(str(r[k]) for k in ('request_kind', 'proposed_kind', 'executed_kind', 'issued_by'))
            for r in missing).items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--local-subset', action='store_true')
    args = ap.parse_args()
    if args.inputs.is_dir():
        manifest = json.loads((args.inputs/'input-manifest.json').read_bytes())
        expected_bytes = (args.inputs/'expected.json').read_bytes()
    else:
        with zipfile.ZipFile(args.inputs) as z:
            manifest = json.loads(z.read('input-manifest.json'))
            expected_bytes = z.read('expected.json')
    assert digest(expected_bytes) == manifest['files']['expected.json']['sha256']
    expected = json.loads(expected_bytes)
    target = {r['episode_id']: r for r in expected['results'] if r['status'] == 'scored'}
    remote = {r['sha256']: str(Path('/archives')/Path(r['path']).relative_to('/opt/colosseum-assurance/research-archives/ethical-scenarios-20260927')) for r in manifest['archives']}
    all_rows, bindings, episodes, versions, skipped = [], [], [], Counter(), []
    seen, duplicate_bindings = {}, []
    for archive in expected['archives']:
        path = Path(archive['archive'] if args.local_subset else remote[archive['sha256']])
        if args.local_subset and not path.is_file():
            skipped.append(archive['sha256'])
            continue
        with zipfile.ZipFile(path) as z:
            names = archive['accessed_members']
            for name in sorted(names):
                if '/episodes/' not in name or not name.endswith('.json'):
                    continue
                episode_id = Path(name).stem
                assert episode_id in target, episode_id
                ledger_name = name.replace('/episodes/', '/privileged_ledgers/')
                ebytes, lbytes = z.read(name), z.read(ledger_name)
                assert digest(ebytes) == names[name] and digest(lbytes) == names[ledger_name]
                record, ledger = json.loads(ebytes), json.loads(lbytes)
                canonical = (digest(json.dumps(record, sort_keys=True).encode()),
                             digest(json.dumps(ledger, sort_keys=True).encode()))
                if episode_id in seen:
                    assert seen[episode_id] == canonical, 'Conflicting duplicate ' + episode_id
                    duplicate_bindings.append({'episode_id': episode_id, 'archive_sha256': archive['sha256'],
                                               'episode_sha256': digest(ebytes), 'ledger_sha256': digest(lbytes)})
                    continue
                seen[episode_id] = canonical
                assert record['episode_id'] == ledger['episode_id'] == episode_id
                assert record['arm_id'] == ledger['arm_id'] == target[episode_id]['arm_id']
                versions[json.dumps(record['code_version'], sort_keys=True)] += 1
                requests = [e for e in ledger['events'] if e['kind'] == 'guard_intervention']
                commands = [e for e in ledger['events'] if e['kind'] == 'command_executed']
                ids = Counter(e['payload'].get('intervention_id') for e in requests)
                assert all(k and n == 1 for k, n in ids.items())
                step_commands, linked, steps = defaultdict(list), defaultdict(list), defaultdict(list)
                for c in commands:
                    step_commands[c['payload'].get('step_index')].append(c)
                    if c['payload'].get('intervention_id'):
                        linked[c['payload']['intervention_id']].append(c)
                for s in record['steps']:
                    steps[s['step_index']].append(s)
                outcomes = target[episode_id]['original_independent_outcome']
                assert len(requests) == outcomes['interventions'] == len(record['interventions'])
                scored = {x['window_id']: x for x in outcomes['measurements']['latencies']
                          if x['metric'] == 'guard_request_to_dispatch'}
                assert len(scored) == len(requests)
                erows = []
                for event in requests:
                    p = event['payload']; identifier = p['intervention_id']; step = p['step_index']
                    matching = linked.get(identifier, [])
                    frozen = scored['guard:' + identifier]
                    assert frozen['status'] == ('observed' if matching else 'right_censored')
                    cs, ss = step_commands.get(step, []), steps.get(step, [])
                    command = cs[0] if len(cs) == 1 else None
                    state = ss[0] if len(ss) == 1 else None
                    cp = command['payload'] if command else {}
                    if command and state:
                        assert abs(command['sim_time_s'] - state['sim_time_s']) < 1e-6
                        assert cp['kind'] == state['executed_command']['kind']
                        assert cp['issued_by'] == state['executed_command']['issued_by']
                    if matching:
                        category = 'exact_link_observed'
                    elif command is None:
                        category = 'no_unique_same_step_command'
                    elif cp.get('issued_by') == 'guard':
                        category = 'guard_command_without_exact_link'
                    elif cp.get('issued_by') == 'controller' and cp.get('kind') == 'land':
                        category = 'retained_controller_landing'
                    elif cp.get('issued_by') == 'controller' and cp.get('kind') == 'hold':
                        category = 'retained_controller_hold'
                    elif cp.get('issued_by') == 'controller' and p['intervention'] == 'suspend_inspection' and cp.get('kind') != 'inspect_capture':
                        category = 'inspection_suspension_noncapture_command'
                    else:
                        category = 'other_retained_command'
                    erows.append({
                        'episode_id': episode_id, 'arm_id': record['arm_id'],
                        'step_index': step, 'request_id': identifier, 'request_time_s': event['sim_time_s'],
                        'request_kind': p['intervention'], 'first_of_kind': p.get('first_of_kind'),
                        'proposed_kind': state['command']['kind'] if state else None,
                        'executed_kind': cp.get('kind'), 'issued_by': cp.get('issued_by'),
                        'command_time_s': command['sim_time_s'] if command else None,
                        'command_intervention_id': cp.get('intervention_id'),
                        'overridden': cp.get('overridden'),
                        'exact_linked_dispatches': len(matching),
                        'frozen_latency_status': frozen['status'],
                        'same_step_command_count': len(cs), 'retained_step_count': len(ss),
                        'guard_state': state.get('provenance', {}).get('guard_state') if state else None,
                        'category': category,
                    })
                all_rows.extend(erows)
                episodes.append({'episode_id': episode_id, 'arm_id': record['arm_id'], **summarize(erows)})
                bindings.append({'archive_sha256': archive['sha256'], 'episode_member': name,
                                 'episode_sha256': digest(ebytes), 'ledger_member': ledger_name,
                                 'ledger_sha256': digest(lbytes)})
    if not args.local_subset:
        assert set(seen) == set(target) and len(seen) == expected['observed_attempts']
    arms = sorted({r['arm_id'] for r in episodes})
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'local_available_subset_validation' if args.local_subset else 'all_recovered_attempts',
        'planned_slots': 792, 'recovered_attempts_in_frozen_analysis': expected['observed_attempts'],
        'processed_attempts': len(seen), 'unrecovered_planned_slots': expected['planned_attempts']-expected['observed_attempts'],
        'expected_sha256': digest(expected_bytes), 'audit_source_sha256': digest(Path(__file__).read_bytes()),
        'validation': {'raw_member_hashes_match': True, 'all_intervention_counts_match': True,
                       'all_exact_linkage_statuses_match': True, 'unique_request_ids': True,
                       'same_step_record_command_fields_match_when_both_present': True},
        'skipped_archives_local_subset_only': skipped,
        'recorded_code_versions': {k: v for k, v in sorted(versions.items())},
        'all': summarize(all_rows),
        'per_arm': {a: summarize([r for r in all_rows if r['arm_id'] == a]) for a in arms},
        'episode_summaries': episodes, 'raw_bindings': bindings, 'request_rows': all_rows,
        'identical_duplicate_bindings': duplicate_bindings,
        'limits': ['Retrospective descriptive mechanism audit; no new flight or causal comparison.',
                   'Same-step commands do not replace exact identifiers in frozen response-time scoring.',
                   'A request is not a distinct fault or a physical command; a retained command is not proof of safe motion.',
                   'Every unresolved planned slot remains unresolved.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'processed_attempts': len(seen), 'all': result['all'],
                      'per_arm': result['per_arm'], 'output_sha256': digest(args.output.read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
