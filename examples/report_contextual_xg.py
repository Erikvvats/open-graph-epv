"""Summarize the frozen contextual teacher and audit source/transfer coverage."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='outputs/contextual_xg_data_v1')
    p.add_argument('--audit',default='outputs/contextual_xg_audit_v1');p.add_argument('--transfer',default='outputs/skillcorner_contextual_xg_v1')
    p.add_argument('--statsbomb',default='data/statsbomb');p.add_argument('--skillcorner',default='data/skillcorner')
    p.add_argument('--out',default='reports');a=p.parse_args();data,run,transfer,out=map(Path,(a.data,a.audit,a.transfer,a.out))
    report=json.loads((run/'report.json').read_text());sc=json.loads((transfer/'report.json').read_text())
    audit=json.loads((data/'audit.json').read_text());wanted=defaultdict(dict)
    for r in audit:wanted[r['match_id']][r['event_id']]=r['context_eligible']
    types=defaultdict(Counter)
    for mid,ids in wanted.items():
        for e in json.loads((Path(a.statsbomb)/f'data/events/{mid}.json').read_text()):
            if e['id'] not in ids:continue
            c=types[e['shot']['type']['name']];goal=int(e['shot']['outcome']['name']=='Goal');eligible=int(ids[e['id']])
            c['shots']+=1;c['goals']+=goal;c['context_eligible']+=eligible;c['eligible_goals']+=goal*eligible
    restarts=Counter()
    for path in sorted((Path(a.skillcorner)/'data/matches').glob('*/*_dynamic_events.csv')):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['event_type']=='player_possession' and r['end_type']=='shot':restarts[r.get('game_interruption_before','')]+=1
    source_audit={'statsbomb_shot_types':dict(types),'skillcorner_shot_preceding_interruption_flags':dict(restarts),
        'interpretation':'Penalty flags are absent on these SkillCorner shot rows; that does not establish that penalties are absent or correctly covered.'}
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    (out/'contextual_xg_v1.json').write_text(json.dumps({'statsbomb':report,'skillcorner_transfer':sc,'source_coverage':source_audit},indent=2)+'\n')
    metrics=report['metrics_on_same_context_shots']
    with (out/'contextual_xg_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','population','n','log_loss','brier','auc','ece_10_equal_width'])
        for name,m in metrics.items():writer.writerow([name,'context_eligible',*[m[k] for k in ('n','log_loss','brier','auc','ece_10_equal_width')]])
        for name,key in [('hybrid','all_shots_hybrid'),('frozen_geometry','all_shots_frozen_geometry')]:
            m=report[key];writer.writerow([name,'all_test_shots',*[m[k] for k in ('n','log_loss','brier','auc','ece_10_equal_width')]])
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    names=['geometry','keeper','context'];colors=['#3498db','#a569bd','#148f77']
    for i,name in enumerate(names):
        loss=metrics[name]['log_loss'];axes[0].bar(i,loss,color=colors[i]);axes[0].text(i,loss+.003,f'{loss:.4f}',ha='center')
    axes[0].set(xticks=range(3),xticklabels=['Geometry','+ Keeper','+ Defenders'],ylim=(0,.32),ylabel='Test log loss (lower is better)',
        title=f"Same {report['n_test_context']:,} shots; 97 held-out-from-fitting matches")
    axes[1].plot([0,1],[0,1],'--',color='gray',lw=1)
    for name,color in [('geometry',colors[0]),('context',colors[2])]:
        bins=metrics[name]['calibration_bins']
        axes[1].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',color=color,label=name)
    axes[1].set(xlim=(0,1),ylim=(0,1.03),xlabel='Mean predicted goal probability',ylabel='Observed goal fraction',title='Calibration: upper bins contain few shots')
    axes[1].legend();fig.savefig(figures/'contextual_xg.png',dpi=160);fig.savefig(figures/'contextual_xg.pdf');plt.close(fig)
    rows='\n'.join(f"| {name} | {m['log_loss']:.6f} | {m['brier']:.6f} | {m['auc']:.4f} | {m['ece_10_equal_width']:.4f} |" for name,m in metrics.items())
    split_rows='\n'.join(f"| {name} | {v['matches']} | {v['shots']} | {v['eligible']} | {v['eligible_goals']} |" for name,v in report['data_audit']['by_split'].items())
    type_rows='\n'.join(f"| {name} | {v['shots']} | {v['context_eligible']} | {v['goals']-v['eligible_goals']} |" for name,v in types.items())
    fallback_rows='\n'.join(f'| {name} | {n} |' for name,n in sc['fallback_reasons'].items())
    c=report['comparisons']['context_minus_geometry'];h=report['all_shots_hybrid'];old=report['all_shots_frozen_geometry']
    improved=100*(1-metrics['context']['log_loss']/metrics['geometry']['log_loss'])
    text=f'''# Kontekstuell StatsBomb-xG og overføring til SkillCorner

**Oppdatering 6. november:** EPV er nå trent og evaluert med denne læreren. Se [det nye EPV-forsøket](contextual_epv.md). Rapporten nedenfor dokumenterer selve xG-forsøket fra 5. november.

Gjennomført 5. november 2026. **Keeper- og forsvarskontekst forbedrer prediksjonen på StatsBomb-testutvalget.** Modellen er trent mot faktiske mål/ikke-mål, med samme kampsplitt som den gamle geometrimodellen. Den nye kontekstmodellen ble valgt på valideringsdata før testevalueringen. Nye SkillCorner-verdier er eksportert separat; eksisterende EPV-modeller og targetfiler er uendret.

## Modell og data

Modellen er L2-regularisert logistisk regresjon med standardisering fra treningsdata. Vi sammenligner åtte geometriske features, disse pluss fem keeperfeatures, og til slutt fem forsvarsfeatures til: **18 features totalt**. Alle tre modellene trenes og sammenlignes på samme kontekstutvalg. Regularisering velges fra 0,1, 1, 10 og 100 etter validerings-log-loss; valgt styrke er henholdsvis 0,1, 10 og 100. Ingen vekter, egenskaper eller terskler er endret etter testresultatene.

Keeperfeatures er avstand til skytter, dybde fra mållinjen, vinkelrett avstand fra skytter–målsentrum-linjen, plassering innenfor skuddtrekanten og avstand til målsentrum. Forsvarsfeatures er nærmeste utespilleravstand, antall forsvarere innen tre meter, antall i skuddtrekanten, antall observerte utespillere og et kontinuerlig nærhetsmål: summen av exp(−avstand/3). Skuddtrekanten går fra skytteren til målstolpene. Dette er geometriske indikatorer, ikke en fysisk blokk-/redningsmodell.

Spillerposisjonene kommer fra `shot.freeze_frame` i StatsBombs hendelsesfiler, ikke fra en ny nedlasting av 360-data. StatsBombs egen [xG-beskrivelse](https://blogarchive.statsbomb.com/articles/soccer/upgrading-expected-goals/) forklarer hvorfor keeper-/forsvarsposisjon og usikkerhet i disse er relevante. Vår implementasjon er en egen, enklere modell og gjenskaper ikke deres proprietære xG.

StatsBomb-koordinater skaleres til en antatt bane på 105 × 68 meter. Modellen bruker ikke `statsbomb_xg`, skuddets sluttposisjon, utfall som input, ballbane etter skuddet, målhøyde, spilleridentitet eller EPV. Kroppsdel og straffetype er utelatt fra denne første overførbare modellen fordi vi ikke har en verifisert tilsvarende SkillCorner-annotasjon. Speiling gir identiske features.

| Del | Kamper | Alle skudd | Brukbar kontekst | Mål i kontekstutvalget |
|---|---:|---:|---:|---:|
{split_rows}

Totalt har 16 314 av 16 509 skudd brukbar kontekst. Vi krever én identifisert motstanderkeeper, minst én utespiller i forsvar, endelige koordinater og skuddposisjon foran mållinjen. Manglende kontekst erstattes ikke med kunstige spillerposisjoner. Råfilene og den eksisterende skuddtabellen verifiseres med SHA256.

## Den viktige straffeskjevheten

De 195 ekskluderte kontekstene inneholder **143 mål**. Av disse er 179 skudd straffer, hvorav 138 mål. Kun én av 180 straffer i korpuset har kvalifiserende kontekst. Kontekstmodellen har derfor nesten ingen støtte for straffesituasjoner. Fravær av keeper/forsvar skal ikke tolkes som «åpent mål».

| Skuddtype | Alle skudd | Brukbar kontekst | Mål uten brukbar kontekst |
|---|---:|---:|---:|
{type_rows}

Denne skjevheten gjør en sammenligning på like utvalg nødvendig. Vi rapporterer også alle testskudd med en hybrid: valgt kontekstmodell når kontekst finnes, ellers den opprinnelige geometrimodellen. Det løser ikke modelleringen av straffer; det sikrer at de ikke forsvinner fra evalueringen. En fremtidig straffemodell trenger en separat, verifisert mapping og et nytt protokollfestet eksperiment.

## Resultater på samme 2 432 testskudd

Utvalget kommer fra 97 kamper som ikke ble brukt til fitting eller valg av regularisering. De samme kampene ble undersøkt i tidligere geometriforsøk, så dette er en utforskende utvidelse av den eksisterende testen, ikke et nytt urørt testsett.

| Modell | Log loss ↓ | Brier ↓ | AUC ↑ | ECE, 10 intervaller ↓ |
|---|---:|---:|---:|---:|
{rows}

`geometry` er refittet på samme kontekstutvalg som de nye modellene. `frozen_geometry` er den gamle modellen trent på alle 11 486 treningsskudd. Keeper og forsvar gir **{improved:.2f} % lavere log loss** enn kontrollen med samme treningsutvalg.

Den primære log-loss-forskjellen er {c['log_loss_difference']:.6f}, med beskrivende 95 % kamp-bootstrap-intervall [{c['descriptive_match_bootstrap_95_interval'][0]:.6f}, {c['descriptive_match_bootstrap_95_interval'][1]:.6f}]. Negativt favoriserer kontekstmodellen. Bootstrap bruker 2 000 parvise trekk av hele kamper, betinget på ferdigtrente modeller. Den inkluderer ikke ny trening eller usikkerhet ved overføring til en annen dataleverandør.

![Prediksjon og kalibrering](figures/contextual_xg.png)

På **alle 2 463 testskudd**, inkludert manglende kontekst, er hybridens log loss **{h['log_loss']:.6f}** mot **{old['log_loss']:.6f}** for gammel geometri. Brier blir {h['brier']:.6f} mot {old['brier']:.6f}, og AUC {h['auc']:.4f} mot {old['auc']:.4f}.

Kalibreringen er ikke entydig bedre: på hele testutvalget er ECE {h['ece_10_equal_width']:.4f}, mot {old['ece_10_equal_width']:.4f} før. ECE avhenger av intervallvalg og små høy-xG-grupper. Bedre log loss og Brier betyr bedre samlede sannsynlighetsprediksjoner på dette utvalget, men er ingen garanti for riktig sannsynlighet i alle situasjonstyper. Straffer og kildeoverføring er særlig uavklarte.

## Overføring til SkillCorner

StatsBombs skuddpunkt og SkillCorners hendelsespunkt brukes som geometrisk anker. SkillCorners hendelseskoordinater er allerede orientert mot angrepsretningen; bare trackingkoordinatene roteres. Keeper og utespillere må være observerte, ikke ekstrapolerte. Vi bruker eksakt `frame_end − 1` (omtrent 0,1 sekund før registrert skudd), et matchet besittelsesendepunkt, observerbar ball/skytter og maksimalt fire meter både mellom skytter og ball og mellom hendelsens skuddpunkt og den trackede skytteren. Dette er kvalitetskontroller, ikke en garanti for perfekt kontaktøyeblikk.

Av 526 skuddmarkører har **{sc['n_context_available']}** tilstrekkelig kontekst. **{sc['n_new_model_applied']}** får ny modell; **{526-sc['n_new_model_applied']}** beholder gammel geometri. Ett ellers kvalifiserende skudd hadde keeperdybde utenfor treningsutvalgets min/maks. Fallback skjer hvis en valgt input er utenfor treningsintervallet. Det beskytter mot noen ekstreme ekstrapolasjoner; å ligge innenfor min/maks er ikke bevis på korrekt kalibrering.

| Årsak til reserve-modell | Antall |
|---|---:|
{fallback_rows}

Gjennomsnittlig estimert xG over alle 526 skudd endres fra **{sc['mean_old_geometry_xg']:.4f}** til **{sc['mean_xg']:.4f}**. Gjennomsnittlig absolutt endring er {sc['mean_absolute_xg_change']:.4f}. Det beskriver hvordan estimatene endres, ikke at sannsynlighetene er validert på SkillCorner.

Fordelingskontrollen viser blant annet større keeperavstand til målsentrum i SkillCorner-kontekstene enn i StatsBomb-treningen, omtrent 0,47 treningsstandardavvik. Vi dokumenterer alle 18 featurefordelingene. Ingen SkillCorner-målflagg brukes til å fitte eller velge xG-modellen.

Ingen av de 526 SkillCorner-skuddradene har `game_interruption_before=penalty_for`. Dette beviser ikke at datagrunnlaget er uten straffer, eller at alle straffer og skudd er korrekt registrert. Skudddekning, straffeidentifikasjon og skuddenes faktisk mål/ikke-mål må derfor fortsatt kontrolleres før påstander om målkalibrering eller taktisk forståelse.

## Bruk og reproduksjon

Fra `projects/open-graph-epv`, med pakken installert eller `PYTHONPATH=src`:

```sh
python -m open_graph_epv.contextual_xg prepare --out outputs/contextual_xg_data_v1
python -m open_graph_epv.contextual_xg select --out outputs/contextual_xg_v1
python -m open_graph_epv.contextual_xg evaluate --out outputs/contextual_xg_audit_v1
python -m open_graph_epv.contextual_xg transfer --out outputs/skillcorner_contextual_xg_v1
python examples/report_contextual_xg.py
```

Kjøringene krever de allerede nedlastede dataene og den frosne `outputs/xg_v1`-modellen fra hoved-README. Bruk nye utdatamapper ved gjentakelse; eksisterende resultater overskrives ikke. Ingen nye store datasett eller biblioteker er lastet ned.

Modellene ligger i `outputs/contextual_xg_v1/models.json`, med låst valg i `selection.json`. Nye eksperimentelle SkillCorner-targets ligger i **`outputs/skillcorner_contextual_xg_v1/shot_xg.csv`**, med begrunnelse og modellkilde per skudd i `shot_audit.json`. CSV-formatet kan leses av eksisterende EPV-pipeline; ingen skudd er fjernet eller satt til null på grunn av manglende kontekst.

Neste EPV-forsøk bør bruke en separat datasettversjon og denne hybridlæreren. Det må evalueres på samme nye target som sine referanser; lavere/høyere EPV-RMSE på en endret target kan ikke sammenlignes direkte med gamle RMSE-tall. Denne kjøringen har ferdigstilt xG-forsøket og overføringskontrollen, ikke trent EPV på nytt eller etablert SkillCorner-kalibrering.

De **46 offline-testene** omfatter lik fysisk scene hos begge leverandører, riktig koordinatrotasjon, speiling og spillerpermutasjon, uavhengighet av mål-/provider-xG-felter, manglende keeper/ball, tidsforskyvning og ujusterte skuddposisjoner. Eksisterende modeller beholdes som referanser.

Se [numerisk rapport](contextual_xg_v1.json), [CSV med metrikker](contextual_xg_metrics.csv) og [PDF-figur](figures/contextual_xg.pdf). Data kommer fra [StatsBomb Open Data](https://github.com/hudl/open-data) og [SkillCorner Open Data](https://github.com/SkillCorner/opendata); eksisterende attribusjon og separate datavilkår i README gjelder fortsatt. Ingen ekstern publisering er utført.
'''
    (out/'contextual_xg.md').write_text(text)
    print('Wrote contextual xG report, source-type audit, transfer audit and PNG/PDF figures.')


if __name__=='__main__':main()
