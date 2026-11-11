"""Report both trained team values and build an offline two-team sequence viewer."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from open_graph_epv.contextual_epv import period_for_frame
from open_graph_epv.diagnostics import choose_cases, scene_records
from open_graph_epv.source import digest, match_path
from open_graph_epv.team_epv import FIELDS, TARGET, load_data, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='outputs/contextual_epv_graphs_v1')
    p.add_argument('--data', default='outputs/team_epv_data_v1')
    p.add_argument('--runs', default='outputs/team_epv_v1')
    p.add_argument('--root', default='data/skillcorner')
    p.add_argument('--out', default='outputs/team_epv_visualizations_v1')
    p.add_argument('--reports', default='reports')
    p.add_argument('--template', default='assets/team_epv_explorer.html')
    args = p.parse_args()
    runs, root, out, reports = map(Path, (args.runs, args.root, args.out, args.reports))
    if out.exists():
        raise FileExistsError('Choose a new visualization output directory')
    arrays, manifest, protocol = load_data(args.base, args.data)
    selection = json.loads((runs/'selection.json').read_text())
    audit = json.loads((runs/'audit/report.json').read_text())
    if (selection['source_sha256'] != source_hashes() or audit['target'] != TARGET
            or audit['selection_sha256'] != digest(runs/'selection.json')
            or audit['predictions_sha256'] != digest(runs/'audit/predictions.npz')):
        raise ValueError('Changed team evaluation')
    for record in selection['models']:
        for name, key in (('model.pt', 'model_sha256'), ('training.json', 'report_sha256')):
            if digest(runs/record['name']/name) != record[key]:
                raise ValueError('Changed model after selection')
    te = np.isin(arrays['match_id'], protocol['split']['test'])
    x, masks, signed, mids, frames, home, y = [arrays[k][te] for k in ('x', 'mask', 'y', 'match_id', 'frame', 'possession_is_home', 'home_away')]
    with np.load(runs/'audit/predictions.npz', allow_pickle=False) as z:
        if not np.array_equal(z['match_id'], mids) or not np.array_equal(z['frame'], frames):
            raise ValueError('Prediction row order changed')
        if not np.array_equal(np.stack([z[f'{f}_target'] for f in FIELDS], -1), y):
            raise ValueError('Team targets changed')
        predictions = {f: np.stack([z[f'{f}_{field}'] for field in FIELDS], -1) for f in ('gnn', 'set', 'ball')}
    metadata = {}
    original = json.loads((Path(args.base)/'manifest.json').read_text())
    for mid in np.unique(mids):
        for suffix in ('match.json', 'tracking_extrapolated.jsonl'):
            path = match_path(root, int(mid), suffix)
            if digest(path) != original['inputs'][str(path.relative_to(root))]:
                raise ValueError('Changed raw source')
        metadata[int(mid)] = json.loads(match_path(root, int(mid), 'match.json').read_text())
    relative_net = {f: (v[:, 0]-v[:, 1])*np.where(home, 1, -1) for f, v in predictions.items()}
    cases = choose_cases(signed, relative_net, mids, frames)
    periods = np.array([period_for_frame(metadata[int(m)], int(f))['period'] for m, f in zip(mids, frames)])
    wanted = set()
    for case in cases:
        i = case['index']
        case['indices'] = np.flatnonzero((mids == mids[i]) & (periods == periods[i]) & (abs(frames-frames[i]) <= 200)).tolist()
        wanted.update(case['indices'])
    scenes = scene_records(root, metadata, x, masks, mids, frames, wanted)
    payload_cases = []
    for case in cases:
        i = case['index'];mid = int(mids[i]);meta = metadata[mid]
        states = []
        for j in case['indices']:
            scene = scenes[j]
            if (scene['sign'] == 1) != home[j]:
                raise ValueError('Raw home/away mapping differs from prepared team labels')
            period = period_for_frame(meta, int(frames[j]))
            states.append({**scene, 'frame': int(frames[j]), 'seconds': float((frames[j]-frames[i])/10),
                'period_seconds': float((frames[j]-period['start_frame'])/10),
                'values': {f: dict(zip(FIELDS, values[j].tolist())) for f, values in predictions.items()},
                'targets': dict(zip(FIELDS, y[j].tolist()))})
        payload_cases.append({'title': case['title'], 'selection': case['selection'] + ' Utvalgskriteriet bruker signert target eller avledet nettofeil; begge lagverdiene vises.', 'match_id': mid,
            'home': meta['home_team']['name'], 'away': meta['away_team']['name'],
            'pitch_length': meta['pitch_length'], 'pitch_width': meta['pitch_width'],
            'anchor_position': case['indices'].index(i), 'states': states})
    payload = {'cases': payload_cases, 'selected_family': selection['selected_family'], 'target': TARGET,
               'metrics': audit['metrics'], 'fields': FIELDS, 'teacher_coverage': manifest['teacher_coverage']}
    template = Path(args.template).read_text()
    if template.count('__EPV_DATA__') != 1:
        raise ValueError('Invalid viewer template')
    out.mkdir(parents=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (out/'explorer.html').write_text(template.replace('__EPV_DATA__', encoded))
    families = ['ball', 'set', 'gnn'];colors = ['#d47727', '#8c59a0', '#167c80']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, key, title in zip(axes, ('home_team_rmse', 'away_team_rmse'), ('Home team EPV', 'Away team EPV')):
        values = [audit['metrics'][f][key] for f in families]
        for j, (v, color) in enumerate(zip(values, colors)):
            ax.bar(j, v, color=color);ax.text(j, v*1.015, f'{v:.5f}', ha='center')
        ax.axhline(audit['training_prior'][key], ls='--', color='#718287', label='Training prior')
        ax.set(xticks=range(3), xticklabels=['Ball only', 'No messages', 'GNN'],
               ylim=(0, max(max(values), audit['training_prior'][key])*1.18), title=title, ylabel='Test RMSE')
        ax.legend()
    fig.suptitle('Two nonnegative team values · 4,228 test snapshots / 3 previously examined matches')
    fig.savefig(out/'team_epv_metrics.png', dpi=160)
    traces, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for ax, case in zip(axes.flat, payload_cases):
        states = case['states'];times = [s['seconds'] for s in states]
        # Insert gaps rather than draw across missing snapshots.
        for field, color, label in zip(FIELDS, ('#178caa', '#e55f58'), ('Home EPV', 'Away EPV')):
            tx, values = [], []
            for j, state in enumerate(states):
                if j and state['frame']-states[j-1]['frame'] > 20:
                    tx.append(np.nan);values.append(np.nan)
                tx.append(state['seconds']);values.append(state['values']['gnn'][field])
            ax.plot(tx, values, 'o-', color=color, ms=2, label=label)
        ax.set(title=case['title'], xlabel='Seconds relative to chosen anchor', ylabel='Expected first-shot xG', ylim=(0, None))
        ax.legend(fontsize=8)
    traces.suptitle('GNN predictions for both teams, simultaneously · hindsight-selected examples')
    traces.savefig(out/'team_epv_sequences.png', dpi=150)
    with PdfPages(out/'team_epv.pdf') as pdf:
        pdf.savefig(fig);pdf.savefig(traces)
    plt.close(fig);plt.close(traces)
    receipt = {'fields': FIELDS, 'target': TARGET, 'selection_sha256': digest(runs/'selection.json'),
        'audit_sha256': digest(runs/'audit/report.json'), 'source_sha256': digest(Path(__file__)),
        'template_sha256': digest(args.template), 'raw_scenes_verified': True, 'raw_home_away_identity_verified': True,
        'n_cases': len(payload_cases), 'n_scene_states': sum(len(c['states']) for c in payload_cases),
        'both_values_nonnegative': all((v >= 0).all() for v in predictions.values()),
        'both_values_positive_n_test': {f: int((v > 0).all(1).sum()) for f, v in predictions.items()}}
    (out/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    reports.mkdir(parents=True, exist_ok=True);(reports/'figures').mkdir(exist_ok=True)
    import shutil
    shutil.copyfile(out/'team_epv_metrics.png', reports/'figures/team_epv_metrics.png')
    (reports/'team_epv_v1.json').write_text(json.dumps({'selection': selection, 'audit': audit, 'visualization': receipt}, indent=2)+'\n')
    rows = '\n'.join(f"| {f} | {selection['validation'][f]['component_rmse']:.6f} | {audit['metrics'][f]['component_rmse']:.6f} | {audit['metrics'][f]['home_team_rmse']:.6f} | {audit['metrics'][f]['away_team_rmse']:.6f} |" for f in families)
    legacy = audit['legacy_net_comparison']
    legacy_text = ('Ingen eldre nettoprediksjoner ble brukt til sekundærsammenligning.' if legacy is None else
        f"Som sekundær kontroll gir differansen mellom GNN-ens nye lagverdier netto-RMSE {audit['metrics']['gnn']['home_net_rmse']:.6f}, mot {audit['legacy_gnn_net_rmse']:.6f} for gammel GNN på identiske targets og situasjoner. Parvis netto-MSE-forskjell er {legacy['mse_difference']:.8f}, med beskrivende 95 % kampintervall [{legacy['ci95'][0]:.8f}, {legacy['ci95'][1]:.8f}]. Dette måler nettoverdien, ikke kvaliteten på de to komponentene hver for seg.")
    report = f'''# To separate EPV-verdier: hjemme og borte

Ny hovedversjon, 10. november 2026. Modellen predikerer **`home_team_epv` og `away_team_epv` separat**. Begge er ikke-negative og kan være positive samtidig. De kommer fra en ny modell med to komponenter; vi klipper ikke den gamle nettoprediksjonen i en positiv og en negativ del.

## Målvariabler

Vi beholder horisonten (t, t + 15 sekunder] og **det første skuddet totalt** i samme omgang. Hjemmetarget er skuddets kontekstuelle xG dersom hjemmelaget tar dette skuddet, ellers null. Bortetarget defineres tilsvarende. Hvis det ikke kommer skudd, er begge null. Dette er ikke første skudd separat for hvert lag, og verdiene summerer ikke xG over flere skudd.

Modellen lærer E[Y_home | situasjon] og E[Y_away | situasjon]. Et bestemt observert vindu har maksimalt én positiv target, men før framtiden er kjent kan begge lagenes forventninger være positive. For eksempel kan 0,03 og 0,02 gi samme nettoverdi som 0,11 og 0,10, selv om situasjonene har ulik samlet forventet skuddverdi.

Netto hjemmeverdi kan fortsatt beregnes som `home_team_epv - away_team_epv`, men er nå en avledet størrelse. Verdiene er forventet første-skudd-xG, ikke direkte sannsynligheten for minst ett mål. Summen er heller ikke sannsynligheten for et skudd.

## Trening og lagidentitet

De samme 29 374 grafene brukes. Vi leser hjemme-/bortelag og faktisk ballbesittelse fra verifiserte råfiler. Nye lagtargets rekonstruerer nøyaktig den gamle signerte targeten som differanse, men treningen minimerer gjennomsnittlig MSE over **begge komponentene**. Ni modeller er trent fra ny initialisering: GNN, spillermodell uten meldinger og ballmodell, tre seeds hver. Samme kampsplitt, maks 30 epoker og tidlig stopp etter åtte epoker uten forbedring brukes.

Internt er input fortsatt orientert etter laget med ballen. Utgangene er forventning for besittelseslaget og motstanderen. API-et bytter kanalene ved behov for å returnere faktisk hjemme- og bortelag; verdiene negeres aldri. Hjemme-/borteidentiteten endres ikke ved besittelsesbytte eller pause.

Modellen har tre softmax-logitter, hvor de to første er lagverdiene og den tredje er restmasse. Dette sikrer ikke-negative verdier med sum høyst én. Restmassen skal ikke tolkes som sannsynligheten for ingen skudd. Bare de to EPV-komponentene inngår i MSE. Speilaugmentering og speilsnitt beholdes.

Læreren er uendret: 413 SkillCorner-skudd med kontekstuell xG og 113 geometriske reserveverdier. Ingen labels fjernes. De nye målfilene lagrer også lag-ID-er og om hjemmelaget hadde ballen, og gjenbruker grafdata uten å kopiere store trackingfiler.

## Resultater

Valideringsvinner: **{selection['selected_family']}**, valgt etter to-komponent-MSE før testberegning. GNN finnes alltid i inferenspakken.

| Modell / tre seeds | Validerings-RMSE, begge | Test-RMSE, begge | Test-RMSE hjemme | Test-RMSE borte |
|---|---:|---:|---:|---:|
{rows}

Konstant referanse basert på treningens forventninger for besittelseslag/motstander har samlet test-RMSE **{audit['training_prior']['component_rmse']:.6f}**. Den mappes til hjemme/borte etter hvem som har ballen. Komponent-RMSE kan ikke sammenlignes direkte med tidligere netto-RMSE, fordi tapsfunksjonen og dimensjonen er forskjellige.

![Separate lagverdier](figures/team_epv_metrics.png)

{legacy_text}

Dette er en utforskende evaluering på tre tidligere undersøkte testkamper. Det etablerer ikke bedre målkalibrering, taktisk forståelse eller overlegenhet over SkillCorner. Eldre SkillCorner-sammenligninger og følsomhetsdiagnostikk gjelder den gamle nettomodellen.

## API og visualisering

```python
from open_graph_epv.team_epv import load_bundle

model = load_bundle("outputs/team_epv_v1/bundle", family="gnn")
result = model.score_frame(current_frame, previous_frame, match_metadata)
home_team_epv = result["home_team_epv"]
away_team_epv = result["away_team_epv"]
```

Resultatet inkluderer faktiske `home_team_id` og `away_team_id`. `net_home_epv` er kun den avledede differansen. `seed_std` gir modellspredning for hver komponent, ikke kalibrerte konfidensintervaller. Utelat `family` for å laste valideringsvinneren.

Åpne lokalt `{out}/explorer.html` i nettleseren. Den nye sekvensviseren har to samtidige verdikort og to kurver, én for hvert lag, med faste lagfarger og fast hjemmelagsorientert bane. Modellbytte, avspilling, tidsskyver og begge realiserte targetkurvene er tilgjengelige. `{out}/team_epv.pdf` inneholder resultat- og sekvensfigurer. Eksemplene er valgt med etterpåinformasjon og er ikke et representativt ytelsesutvalg.

Reproduser etter klargjøring av de kontekstuelle basisgrafene:

```bash
python -m open_graph_epv.team_epv prepare --out outputs/team_epv_data_v1
python examples/run_team_epv.py --legacy ""
python examples/report_team_epv.py
```

Bruk nye utdatamapper hvis kjøringene finnes. `--legacy ""` gjør at en ny installasjon ikke trenger å trene den gamle nettomodellen; uten flagget kan dens eksisterende audit brukes til sekundærsammenligningen. Se [full numerisk rapport](team_epv_v1.json) og hoved-README for datanedlasting og xG-læreren. SkillCorner-/StatsBomb-attribusjon og separate datavilkår gjelder fortsatt.
'''
    (reports/'team_epv.md').write_text(report)
    print(f'Wrote two-team viewer, PDF, figures and {reports}/team_epv.md')


if __name__ == '__main__':
    main()
