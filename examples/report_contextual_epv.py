"""Publish the frozen contextual-target EPV evaluation without model refitting."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from open_graph_epv.contextual_epv import TARGET, verify_data
from open_graph_epv.evaluation import verify_selection
from open_graph_epv.source import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='outputs/contextual_epv_graphs_v1')
    parser.add_argument('--runs', default='outputs/contextual_epv_v1')
    parser.add_argument('--provider', default='outputs/contextual_provider_v1/audit/report.json')
    parser.add_argument('--out', default='reports')
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    manifest = verify_data(args.data)
    selection = json.loads((runs / 'selection.json').read_text())
    audit = json.loads((runs / 'audit/report.json').read_text())
    verify_selection(args.data, runs, selection)
    if audit.get('target') != TARGET or audit['selection_sha256'] != digest(runs / 'selection.json'):
        raise ValueError('Audit and selected contextual models differ')
    provider_path = Path(args.provider)
    provider = json.loads(provider_path.read_text()) if provider_path.exists() else None
    if provider is not None and provider.get('target') != TARGET:
        raise ValueError('Provider comparison uses a different target')
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    report = {'date': '2026-11-06', 'target': TARGET, 'dataset': manifest,
              'selection': selection, 'audit': audit, 'provider_comparison': provider}
    (out / 'contextual_epv_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    names = {'mean': 'Treningsgjennomsnitt', 'ball': 'Bare ball',
             'set': 'Spillere uten meldinger', 'gnn': 'GNN'}
    val = {k: v['ensemble']['rmse'] for k, v in selection['validation'].items()}
    val['mean'] = selection['validation']['gnn']['ensemble']['baseline_rmse']
    test = {k: v['ensemble']['rmse'] for k, v in audit['models'].items()}
    test['mean'] = audit['training_mean_reference']['rmse']
    with (out / 'contextual_epv_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'validation_rmse', 'test_rmse', 'target'])
        for family in names:
            writer.writerow([family, val[family], test[family], TARGET])
    rows = '\n'.join(f'| {name} | {val[k]:.6f} | {test[k]:.6f} |' for k, name in names.items())
    comparisons = '\n'.join(
        f"| {name} | {c['mse_difference']:.8f} | [{c['ci95'][0]:.8f}, {c['ci95'][1]:.8f}] |"
        for name, c in audit['comparisons'].items())
    subsets = '\n'.join(
        f"| {name} | {s['n']} | " + ' | '.join(f"{s['models'][k]['rmse']:.6f}" for k in ('gnn', 'ball', 'set')) + ' |'
        for name, s in audit['test_target_subsets'].items() if s['n'])
    seeds = '\n'.join(
        f"| {family} | {v['seed']} | {v['rmse']:.6f} | {t['rmse']:.6f} |"
        for family in ('gnn', 'ball', 'set')
        for v, t in zip(selection['validation'][family]['individual'], audit['models'][family]['individual']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    order = list(names)
    english = ['Training mean', 'Ball only', 'No messages', 'GNN']
    colors = ['#85929e', '#3498db', '#a569bd', '#148f77']
    for ax, values, title in zip(axes, (val, test), ('Validation: 3 matches', 'Exploratory test: 3 matches')):
        for index, family in enumerate(order):
            ax.bar(index, values[family], color=colors[index])
            ax.text(index, values[family] + max(values.values()) * .012, f'{values[family]:.5f}', ha='center', fontsize=9)
        ax.set(xticks=range(4), xticklabels=english, ylim=(0, max(values.values()) * 1.15),
               ylabel='RMSE against signed future contextual xG', title=title)
    fig.suptitle('Same target and snapshots for all models; 3 seeds per learned family')
    fig.savefig(figures / 'contextual_epv.png', dpi=160)
    fig.savefig(figures / 'contextual_epv.pdf')
    plt.close(fig)
    provider_text = 'Den oppdaterte leverandørsammenligningen er ikke generert i denne rapporten.'
    if provider is not None:
        provider_rows = '\n'.join(f"| {name} | {m['rmse']:.6f} |" for name, m in provider['metrics'].items())
        provider_contrasts = '\n'.join(
            f"| {name} | {provider['comparisons'][name]['mse_difference']:.8f} | "
            f"[{provider['comparisons'][name]['ci95'][0]:.8f}, {provider['comparisons'][name]['ci95'][1]:.8f}] |"
            for name in ('gnn_minus_skillcorner_adapted', 'gnn_adapted_minus_skillcorner_adapted'))
        provider_text = f'''Her evalueres **{provider['n_test']} event-start-situasjoner fra {provider['n_test_matches']} testkamper**, med {provider['calibration']['n_validation']} valideringssituasjoner til tilpasning. Det er et annet utvalg enn snapshot-tabellen over. Alle verdier måles mot den nye kontekstuelle målvariabelen. SkillCorners for-/motverdier tilpasses målvariabelen med en affin modell tilpasset kun valideringskampene; råverdier og en tilsvarende tilpasset GNN rapporteres også.

| Modell / referanse | Test-RMSE |
|---|---:|
{provider_rows}

| Parvis MSE-forskjell | Forskjell | Beskrivende 95 % kampintervall |
|---|---:|---:|
{provider_contrasts}

SkillCorners måldefinisjon og horisont avviker fra vår. Resultatene tester hvor godt verdiene predikerer vår lærerdefinerte målvariabel, og dokumenterer ikke bedre prediksjon av faktiske mål eller overlegenhet over leverandørens produkt. Modellversjon og eventuell overlapp med leverandørens trening er ukjent. Parvise kampintervaller og antall observasjoner finnes i den numeriske rapporten.'''
    text = f'''# EPV med kontekstuell xG som målvariabel

Hovedprosjektet bruker nå [to separate lagverdier](team_epv.md): `home_team_epv` og `away_team_epv`. Rapporten nedenfor gjelder den tidligere nettomodellen.

Ferdig trent og evaluert 6. november 2026. **Dette dokumenterer den tidligere modellen med én signert nettoverdi.** Ni nye modeller er trent på SkillCorner-tracking med målverdier fra vår StatsBomb-trente kontekstuelle xG-modell. Modellen valgt på valideringsdata er **{selection['selected_family']}**. GNN-en beholdes i inferenspakken uansett valideringsvinner.

## Hva modellen skal lære

For en situasjon s lærer modellen den betingede forventningen E[Y | s]. Y er xG for det første skuddet i de neste 15 sekundene: positivt når laget med ballen nå skyter, negativt når motstanderen skyter, og null hvis det ikke kommer et skudd. Skudd i selve snapshot-øyeblikket er ekskludert; skudd nøyaktig 15 sekunder senere er inkludert. Hele horisonten må ligge innen samme omgang. Klokken løper også under avbrudd, og målverdien er ikke begrenset til samme ballbesittelse.

Skuddets xG beregnes med skuddgeometri, keeperposisjon og forsvarerposisjoner. Den separate læreren er logistisk regresjon med 18 features, trent mot faktiske mål/ikke-mål i StatsBomb. Den er **ikke StatsBombs proprietære xG-modell**. Se [xG-eksperimentet](contextual_xg.md) for features, validering og kildeoverføring.

Fremtidig skuddkontekst brukes bare til å lage treningsmålet. EPV får nåværende tracking og bakoverberegnet hastighet som input, uten fremtidig skuddinformasjon. MSE brukes fordi optimum i forventning er den betingede middelverdien. En robust tapsfunksjon som Smooth L1 har generelt et annet optimum og kan nedvekte sjeldne, verdifulle skudd.

Verdien er **forventet signert første-skudd-xG**, ikke direkte sannsynligheten for minst ett mål. Koblingen til målrisiko avhenger av lærerens kalibrering i relevante situasjoner. Flere skudd, returer, den negative motstanderkomponenten og den valgte horisonten gjør at disse begrepene ikke kan brukes om hverandre.

## Datagrunnlag og dekning

StatsBomb-læreren bruker 11 346 kontekstuelle treningsskudd; 2 536 valideringsskudd velger modell/regularisering, og 2 432 testskudd evaluerer kontekstmodellene på samme utvalg. På disse testskuddene er log loss 0,258118 mot 0,275918 for en geometrikontroll trent på samme utvalg. Dette er evaluering på StatsBomb, ikke validering av målkalibrering i SkillCorner.

SkillCorner bidrar med 20 kamper og 526 skuddmarkører. **413 skudd får kontekstuell xG; 113 bruker den dokumenterte geometriske reservemodellen.** Alle skudd beholdes. Manglende kontekst blir aldri gjort om til null eller til «ingen skudd». Ett ellers tilgjengelig kontekstskudd faller utenfor lærerens treningsintervall og får reserveverdi.

Dette er derfor en hybrid målvariabel med kontekstuell xG som hovedkilde. En modell trent utelukkende på kontekstuelle targets ville kreve at vinduer med ukjent target ble ekskludert og at utvalgsskjevheten ble vurdert separat. Vi velger full skudddekning med eksplisitt provenance.

De 29 374 grafene fordeles på 20 862 trening, 4 284 validering og 4 228 test, fra henholdsvis 14/3/3 hele kamper. Input, maske, kamp, frame og skuddtilstedeværelse er kontrollert identiske med tidligere snapshot-datasett. Det som endres, er skuddverdiene. Totalt har 26 509 vinduer ingen fremtidige skudd, 2 350 har kontekstuell xG og 515 har reserve-xG. Flere vinduer kan vise mot samme skudd og er ikke uavhengige observasjoner.

## Modell og forsøksoppsett

GNN-en har to lag med meldingsutveksling og 31 537 parametere. Grafen inneholder ballen og opptil 22 spillere. Input er posisjon, bakoverberegnet hastighet, lagtilhørighet, keeper-/ballindikatorer og observasjonsmasker. Ekstrapolerte noder maskeres. Nåværende besittelseslag angriper alltid mot +x.

Kontrollene er en ballmodell og samme spillermodell uten meldingsutveksling. Hver familie trenes med tre seeds (0, 1, 2), maksimalt 30 epoker og tidlig stopp etter åtte epoker uten bedre validerings-MSE. Alle bruker samme målvariabel og kampsplitt. Familien velges etter MSE for gjennomsnittet av tre modeller på valideringsdata. Kildekode, data og vekter fryses før testberegningen.

Speiling over banens lengdeakse brukes under trening. Inferens er gjennomsnittet av original og speilet situasjon, slik at sidevalg alene ikke endrer verdien. Denne symmetrien gjelder modellens forenklede input; modellen kjenner ikke spillerens foretrukne fot eller identitet.

## Resultater

| Modell / tre-seed-ensemble | Validerings-RMSE | Test-RMSE |
|---|---:|---:|
{rows}

GNN har {100 * audit['models']['gnn']['ensemble']['rmse_skill']:.2f} % lavere test-RMSE enn treningsgjennomsnittet. Valideringsvinneren `{selection['selected_family']}` har {100 * audit['models'][selection['selected_family']]['ensemble']['rmse_skill']:.2f} % lavere test-RMSE enn samme referanse. Sammenligningen med spillermodellen uten meldinger isolerer nytten av meldingsutveksling; den må vurderes sammen med kampintervallene nedenfor.

![Samme kontekstuelle target for alle modellene](figures/contextual_epv.png)

Dette er en utforskende evaluering på tre kamper som også er undersøkt i tidligere eksperimenter. De er holdt utenfor fitting og modellvalg i denne kjøringen, men er ikke et nytt urørt testsett. **Gamle og nye RMSE-tall kan ikke sammenlignes direkte fordi targetverdiene er endret.**

| Parvis MSE-forskjell | Forskjell | Beskrivende 95 % kampintervall |
|---|---:|---:|
{comparisons}

Negativ forskjell favoriserer GNN. Intervallene bruker alle 27 ordnede bootstrap-trekk av tre hele kamper. De er beskrivende og inkluderer verken ny trening eller lærerens usikkerhet. Tre kamper gir ikke grunnlag for sterke generaliseringspåstander.

### Diagnostikk etter targetkilde

| Targettype i test | Vinduer | GNN-RMSE | Ball-RMSE | Uten meldinger-RMSE |
|---|---:|---:|---:|---:|
{subsets}

Gruppene bestemmes av hva som skjer senere. De brukes til diagnostikk etter evaluering, ikke til valg av modell ved inferens. Feilnivåene mellom gruppene kan ikke tolkes som et kontrollert estimat av kontekstens effekt: situasjoner og targetfordelinger er forskjellige.

### Enkeltkjøringer

| Familie | Seed | Validerings-RMSE | Test-RMSE |
|---|---:|---:|---:|
{seeds}

## Sammenligning med SkillCorner

{provider_text}

## Bruk og begrensninger

Se [hoved-README](../README.md) for komplette kommandoer. Den trente pakken ligger lokalt i `outputs/contextual_epv_v1/bundle`. Den inneholder GNN-en, valideringsvinneren, xG-læreren, rapporter, kildekode og sjekksummer. Last med `open_graph_epv.contextual_epv.load_bundle`; `family="gnn"` velger GNN eksplisitt.

Skudddekning, kontaktøyeblikk, straffeklassifisering og overført xG-kalibrering må fortsatt valideres. Nesten alle StatsBomb-straffer mangler kvalifiserende kontekst. Denne modellen er et reproduserbart forskningsprosjekt, ikke dokumentasjon på at den forstår fotball kausalt, anbefaler riktige løp eller gir et validert RL-belønningssignal.

Se [full numerisk rapport](contextual_epv_v1.json), [CSV](contextual_epv_metrics.csv), [PDF-figur](figures/contextual_epv.pdf) og den forhåndsregistrerte [protokollen](../configs/contextual_epv_v1.json). Kilde- og datavilkårene i README gjelder. Ingen ekstern publisering er utført.
'''
    (out / 'contextual_epv.md').write_text(text)
    print('Wrote contextual EPV report, JSON, CSV and PNG/PDF figures.')


if __name__ == '__main__':
    main()
