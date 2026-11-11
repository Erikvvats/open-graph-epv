"""Create the reproducible architecture/history report from the completed audit."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='outputs/attention_history_v1')
    p.add_argument('--out',default='reports');a=p.parse_args();run,out=Path(a.run),Path(a.out)
    report=json.loads((run/'audit/report.json').read_text());selection=report['selection'];protocol=selection['protocol']
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    runs={f'{mode}_seed{seed}':json.loads((run/f'{mode}_seed{seed}'/'training.json').read_text()) for mode in protocol['modes'] for seed in protocol['seeds']}
    compact={**report,'training_runs':{name:{k:r[k] for k in ('mode','seed','parameter_count','best_validation','elapsed_seconds','model_sha256','versions')} for name,r in runs.items()}}
    (out/'attention_history_v1.json').write_text(json.dumps(compact,indent=2)+'\n')
    names=['gnn','snapshot','history','repeat','ball','set']
    labels={'gnn':'GNN','snapshot':'Transformer: nåbilde','history':'Transformer: historikk','repeat':'Transformer: gjentatt nåbilde','ball':'Bare ball','set':'Spillere uten meldinger'}
    with (out/'attention_history_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['family','parameters','validation_rmse','test_rmse','test_mse','test_mae'])
        for name in names:
            m=report['metrics'][name]['ensemble'];v=selection['validation'][name]['ensemble']
            parameters=protocol['gnn_parameters'] if name=='gnn' else protocol['transformer_parameters'] if name in protocol['modes'] else ''
            writer.writerow([name,parameters,v['rmse'],m['rmse'],m['mse'],m['mae']])
    colors=['#117a65','#2874a6','#af601a','#7d3c98']
    shown=names[:4];fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for i,name in enumerate(shown):
        val=selection['validation'][name]['ensemble']['rmse'];test=report['metrics'][name]['ensemble']['rmse']
        axes[0].bar(i-.18,val,width=.36,color=colors[i],alpha=.4)
        axes[0].bar(i+.18,test,width=.36,color=colors[i])
        axes[0].text(i+.18,test+.00045,f'{test:.5f}',ha='center',fontsize=8)
    axes[0].set(xticks=range(4),xticklabels=['GNN','Snapshot\ntransformer','History\ntransformer','Repeated\nsnapshot'],
        ylabel='RMSE: signed future first-shot xG',ylim=(0,.036),title='Three-seed ensembles; light=validation, dark=test')
    comparisons=['snapshot_minus_gnn','history_minus_snapshot','history_minus_repeat']
    for i,key in enumerate(comparisons):
        c=report['comparisons'][key];d=c['mse_difference'];lo,hi=c['ci95']
        axes[1].errorbar(d*1e6,i,xerr=np.array([[d-lo],[hi-d]])*1e6,fmt='o',capsize=5,color=colors[i+1])
    axes[1].axvline(0,color='gray',ls='--',lw=1)
    axes[1].set(yticks=range(3),yticklabels=['Snapshot − GNN','History − snapshot','History − repeat'],
        ylim=(-.5,2.5),xlabel='MSE difference × 10⁶ (negative favors first model)',title='Descriptive whole-match bootstrap intervals')
    fig.suptitle('Exploratory architecture/history comparison: same 4,228 test snapshots',fontsize=12)
    fig.savefig(figures/'attention_history.png',dpi=160);fig.savefig(figures/'attention_history.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,mode in zip(axes,protocol['modes']):
        for seed in protocol['seeds']:
            history=runs[f'{mode}_seed{seed}']['history']
            ax.plot([r['epoch'] for r in history],[r['rmse'] for r in history],label=f'Seed {seed}')
        ax.set(title=mode,xlabel='Epoch',ylabel='Validation RMSE');ax.legend(fontsize=8)
    fig.savefig(figures/'attention_learning_curves.png',dpi=150);plt.close(fig)
    rows='\n'.join(f"| {labels[n]} | {selection['validation'][n]['ensemble']['rmse']:.6f} | {report['metrics'][n]['ensemble']['rmse']:.6f} |" for n in names)
    contrast_rows='\n'.join(f"| {key} | {v['mse_difference']:+.3e} | [{v['ci95'][0]:+.3e}, {v['ci95'][1]:+.3e}] |" for key,v in report['comparisons'].items())
    seed_rows='\n'.join(f"| {mode} | {seed} | {runs[f'{mode}_seed{seed}']['best_validation']['epoch']} | {len(runs[f'{mode}_seed{seed}']['history'])} | {next(v for v in report['metrics'][mode]['individual'] if v['seed']==seed)['rmse']:.6f} |" for mode in protocol['modes'] for seed in protocol['seeds'])
    winner=labels[selection['selected_family']]
    text=f'''# GNN, graph transformer og historikk

Eksperiment gjennomført 5. november 2026. **Modell valgt på valideringsdata: {winner}.** Dette er en utforskende sammenligning på de tidligere brukte testkampene; ingen ny, uavhengig sluttest er opprettet.

## Hva som er sammenlignet

1. Den eksisterende message-passing-GNN-en på ett øyeblikksbilde.
2. En liten graph transformer på samme øyeblikksbilde.
3. Den samme transformeren med spillerhistorikk ved −2, −1, −0,5 og 0 sekunder.
4. En kontroll der nåbildets verdier gjentas på historikktidspunktene, med samme historiske synlighetsmaske for dagens synlige spillere.

Ballmodellen og spillermodellen uten meldinger er beholdt som referanser. De opprinnelige GNN-, ball- og set-modellene er uendret; ni nye transformer-modeller er trent, tre tilfeldige frø per betingelse.

Alle bruker samme StatsBomb-trente geometriske xG-lærer, samme 526 skuddmarkører, samme signerte første-skudd-xG innen 15 sekunder og samme kampsplitt. Datasettet inneholder 20 862 treningsbilder fra 14 kamper, 4 284 valideringsbilder fra tre kamper og 4 228 testbilder fra tre kamper. Null betyr ingen skudd innen horisonten. Et skudd akkurat på nåtidspunktet er utelatt fra fremtidsvinduet; periodens slutt sensurerer ufullstendige horisonter.

## Kapasitet og arkitektur

GNN-en har **{protocol['gnn_parameters']:,} parametere**. Hver transformer har **{protocol['transformer_parameters']:,}**, bare {100*(protocol['transformer_parameters']/protocol['gnn_parameters']-1):.2f} % flere. Transformeren har to romlige attention-lag, fire hoder og 40 skjulte dimensjoner. Den bruker samme nodefeatures, fire relasjonsfeatures, separate gjennomsnitt for lag og ball, samt tanh-utgang som GNN-en. GNN-en har 48 skjulte dimensjoner og aggregerer lærte meldinger med gjennomsnitt.

Attention beregner vekter fra query/key-likhet og en lært bias fra relativ posisjon, avstand og lagtilhørighet. Romlig attention inkluderer egen node. Trajektorie-encoderen bruker nåtidens node som query over samme spillers tidligere representasjoner. Q/K/V- og utgangsprojeksjonene deles med første romlige blokk for å holde parameterbudsjettet fast. Deretter beregnes samspillet mellom spillerne ved nåtidspunktet. Dette er en liten, egen forsøksmodell, ikke hovedprosjektets større GraphTemporalTransformer.

Alle tre transformerbetingelsene har identisk arkitektur og parameterantall. Nåbildevarianten bruker ett tidssteg; tidsencoderen får da null og gir ingen historisk informasjon. Historikkmodellen er ikke en full modell av alle tidligere samspill mellom spillere: den oppsummerer hver spillers egen bane før den romlige modelleringen. Attention-vekter er ikke kausale forklaringer.

## Kontroll mot lekkasje og feil spillerkobling

Historikken er hentet fra eksakte tidligere frames. Vi bruker nåtidens spiller-ID-er og retning relativt til laget som har ballen nå. En annen spillerrekkefølge eller et tidligere bytte av ballbesittelse skal derfor ikke flytte historikken til feil spiller eller snu banen. Tidligere spillere som ikke lenger finnes i nåbildets spillerliste, tas ikke inn som nye noder. Manglende historikk maskeres; vi henter aldri frames fra en annen periode. Hastigheter bruker bare det umiddelbart foregående råbildet.

Nåbildet i hver av de 29 374 historikkradene er kontrollert **bit-for-bit identisk** med det opprinnelige datasettet, inkludert synlighetsmasken. Målvariabelen og radrekkefølgen er uendret. Historiske bilder uten observerbar ball eller tilstrekkelig grafkontekst maskeres konservativt. Den ekstra komprimerte historikkfilen er rundt 19 MB; rådataene er gjenbrukt.

Gjenta-nåbildet-kontrollen beholder tidligere synlighet for spillerne som er synlige nå. Dermed sammenligner historikk mot kontroll primært verdiene i tidligere posisjoner/hastigheter, betinget på samme dekning. Historikk kontra ett nåbilde kan også påvirkes av synlighetsinformasjon og beregningsstruktur.

## Trening og modellvalg

Alle nye kjøringer bruker MSE, AdamW med læringsrate 0,001 og weight decay 0,001, batchstørrelse 64, gradientklipping 1,0 og maksimalt 30 epoker. Tidlig stopp har tålmodighet på åtte epoker. Dette følger den eksisterende GNN-oppskriften; hver arkitektur er ikke separat finjustert til sitt beste mulige oppsett. Det velges checkpoint utelukkende på validerings-MSE.

Speiling under trening bruker samme fortegn for hele historikken til et bilde. Inferens tar speilsnitt og deretter likt gjennomsnitt av frø 0, 1 og 2. Kandidaten blant GNN, nåbilde-transformer og historikk-transformer velges på valideringsfeilen til ensemblet. Gjenta-kontrollen er bare en diagnose. Protokoll, kode, datasett og modellfiler kontrolleres med SHA256; modellvalget lagres før samlet testevaluering. En separat rådata-inferenskontroll under trening beregnet én prediksjon uten å bruke målverdien eller gjøre modelltilpasning.

## Resultater

Lavere RMSE er bedre. Alle radene gjelder de samme situasjonene.

| Modell | Validering RMSE | Test RMSE |
|---|---:|---:|
{rows}

Nåbilde-transformeren vant den forhåndsbestemte valideringsseleksjonen. På test hadde GNN lavest feil av disse arkitekturene: nåbilde-transformerens RMSE er {100*(report['metrics']['snapshot']['ensemble']['rmse']/report['metrics']['gnn']['ensemble']['rmse']-1):.2f} % høyere. Historikk reduserer RMSE med {100*(1-report['metrics']['history']['ensemble']['rmse']/report['metrics']['snapshot']['ensemble']['rmse']):.2f} % mot ett nåbilde og {100*(1-report['metrics']['history']['ensemble']['rmse']/report['metrics']['repeat']['ensemble']['rmse']):.2f} % mot gjenta-kontrollen. Den er fortsatt litt svakere enn GNN samlet.

Historikk slår gjenta-kontrollen på alle tre kampene, og det beskrivende bootstrap-intervallet for akkurat denne kontrasten ligger under null. Dette er et lite positivt signal for ekte trajektorieinformasjon i denne arkitekturen. Kontrastene nåbilde-transformer mot GNN og historikk mot ett nåbilde har intervaller som krysser null. Vi kan derfor ikke hevde en tydelig generell gevinst ved å bytte fra GNN til transformer eller ved å legge til historikk. Modellvalget i de lagrede artifaktene forblir valideringsvalget; ingen ny tuning er gjort etter testresultatene.

![Resultater og usikkerhet](figures/attention_history.png)

Testmålene og utvalget er det opprinnelige snapshot-utvalget. Tallene skal ikke direkte sammenlignes med RMSE fra SkillCorner-rapportens 1 850 hendelsesstarter: den rapporten bruker et annet utvalg av situasjoner.

Planlagte kontraster er MSE for første modell minus MSE for andre. Negativt tall favoriserer første modell. Intervallene er beskrivende hel-kamp-bootstrap med alle 27 ordnede trekk fra tre kamper.

| Kontrast | MSE-forskjell | Beskrivende 95 % intervall |
|---|---:|---:|
{contrast_rows}

Tre kamper er for lite til sterke generaliseringspåstander. Overlappende snapshots er ikke uavhengige observasjoner. Intervallene inkluderer ikke usikkerhet fra ny modelltrening eller overføring av xG-læreren. Testkampene er allerede undersøkt i tidligere eksperimenter, så intervaller som eventuelt ikke krysser null er fortsatt utforskende funn.

| Betingelse | Frø | Valgt epoke | Kjørte epoker | Test RMSE |
|---|---:|---:|---:|---:|
{seed_rows}

![Valideringskurver](figures/attention_learning_curves.png)

## Reproduser

Fra `projects/open-graph-epv`, med pakken installert eller `PYTHONPATH=src`:

```sh
python -c "from open_graph_epv.history_data import prepare; prepare('data/skillcorner', 'outputs/graphs_v1', 'outputs/history_v1')"
python examples/run_attention_history.py --workers 2
python examples/report_attention_history.py
python examples/score_attention.py --mode history --match 2015213 --frame 2240
```

Kommandoene forutsetter at xG-/GNN-artifaktene fra hoved-README er laget. Eksisterende utdata overskrives ikke. Protokollen ligger i `configs/attention_history_v1.json`. Når et nytt datasett eller kode endres, må en ny protokoll registreres og nye utdatamapper brukes.

Ferdige kjøringer kan inspiseres i `outputs/attention_history_v1/<betingelse>_seed<frø>/training.json`. En kontrollert ny kjøring kan startes med `python -m open_graph_epv.attention_experiment train --mode snapshot --seed 0 --out outputs/ny_kjoring`. Modellvalget ligger i `selection.json`, testresultatene i `audit/report.json`, og per-situasjonsprediksjonene i `audit/predictions.npz`. Numerisk rapport for deling: [attention_history_v1.json](attention_history_v1.json). Figuren finnes også som [PDF](figures/attention_history.pdf).

De 36 offline-testene dekker blant annet spillerpermutasjon, speilsymmetri, maskerte verdier, endelige gradienter, likt parameterbudsjett, kontrollenes uavhengighet av tidligere posisjoner, spiller-ID-kobling, besittelsesretning, perioder og fremtidsgrenser. Ingen rådata eller modeller er publisert eksternt av denne kjøringen.
'''
    (out/'attention_history.md').write_text(text)
    print('Wrote attention/history report, metrics, learning curves and PNG/PDF comparison.')


if __name__=='__main__':main()
