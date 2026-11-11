'use strict';
// Display frozen predictions only. This file does not implement model inference.
const data = window.EPV_DEMO;
const fields = ['home_team_epv', 'away_team_epv'];
const colors = ['#178caa', '#e55f58'];
const element = id => document.getElementById(id);
let caseIndex = 0;
let frameIndex = 0;
let timer = null;

function svgNode(parent, tag, attributes, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function drawPitch(sequence, state) {
  const svg = element('pitch');
  svg.replaceChildren();
  const scale = Math.min(740 / sequence.pitch_length, 470 / sequence.pitch_width);
  const px = x => 400 + x * scale;
  const py = y => 270 - y * scale;
  const length = sequence.pitch_length;
  const width = sequence.pitch_width;
  const markings = {fill: 'none', stroke: '#88a091', 'stroke-width': 1.5};
  svgNode(svg, 'rect', {...markings, x: px(-length/2), y: py(width/2), width: length*scale, height: width*scale});
  svgNode(svg, 'line', {...markings, x1: px(0), y1: py(width/2), x2: px(0), y2: py(-width/2)});
  svgNode(svg, 'circle', {...markings, cx: px(0), cy: py(0), r: 9.15*scale});
  for (const side of [-1, 1]) {
    const start = side === -1 ? -length/2 : length/2-16.5;
    svgNode(svg, 'rect', {...markings, x: px(start), y: py(20.16), width: 16.5*scale, height: 40.32*scale});
    svgNode(svg, 'line', {...markings, 'stroke-width': 4, x1: px(side*length/2), x2: px(side*length/2), y1: py(-3.66), y2: py(3.66)});
  }
  svgNode(svg, 'text', {x: 400, y: 22, 'text-anchor': 'middle', fill: '#52676c', 'font-size': 14}, 'Home attacks right throughout');
  // Draw the ball last so a nearby player does not hide it.
  for (const point of [...state.points].sort((a, b) => Number(a.ball)-Number(b.ball))) {
    const x = px(point.x), y = py(point.y), radius = point.ball ? 5 : 9;
    const attributes = {fill: point.ball ? '#dca62f' : colors[point.home ? 0 : 1], stroke: 'white', 'stroke-width': 1.5};
    if (point.keeper) svgNode(svg, 'rect', {...attributes, x: x-radius, y: y-radius, width: radius*2, height: radius*2});
    else svgNode(svg, 'circle', {...attributes, cx: x, cy: y, r: radius});
    if (!point.ball) svgNode(svg, 'text', {x, y: y+3, fill: 'white', 'font-size': 9, 'text-anchor': 'middle'}, point.number);
  }
}

function drawTimeline(sequence, family) {
  const svg = element('timeline');
  svg.replaceChildren();
  const states = sequence.states;
  const showTargets = element('targets').checked;
  const values = states.flatMap(s => fields.map(k => s.values[family][k]));
  if (showTargets) values.push(...states.flatMap(s => fields.map(k => s.targets[k])));
  const upper = Math.max(0.002, ...values)*1.12;
  const start = states[0].seconds, end = states[states.length-1].seconds;
  const px = t => 65 + (t-start)/Math.max(1, end-start)*710;
  const py = v => 212-v/upper*190;
  for (let i = 0; i <= 4; i++) {
    const value = upper*i/4, y = py(value);
    svgNode(svg, 'line', {x1: 65, x2: 775, y1: y, y2: y, stroke: '#dbe5e0'});
    svgNode(svg, 'text', {x: 55, y: y+4, 'text-anchor': 'end', fill: '#52676c', 'font-size': 12}, value.toFixed(3));
  }
  for (let i = 0; i <= 4; i++) {
    const t = start+(end-start)*i/4;
    svgNode(svg, 'text', {x: px(t), y: 240, 'text-anchor': 'middle', 'font-size': 12, fill: '#52676c'}, `${t.toFixed(1)} s`);
  }
  function curve(field, color, target) {
    let path = '';
    states.forEach((state, i) => {
      const value = target ? state.targets[field] : state.values[family][field];
      const gap = i === 0 || state.frame-states[i-1].frame > 20;
      path += `${gap ? 'M' : 'L'} ${px(state.seconds)} ${py(value)} `;
    });
    svgNode(svg, 'path', {d: path, fill: 'none', stroke: color, 'stroke-width': target ? 1.5 : 2.8, 'stroke-dasharray': target ? '5 4' : 'none'});
    for (const state of states) {
      const value = target ? state.targets[field] : state.values[family][field];
      svgNode(svg, 'circle', {cx: px(state.seconds), cy: py(value), r: target ? 2 : 3, fill: color});
    }
  }
  if (showTargets) fields.forEach((field, i) => curve(field, colors[i], true));
  fields.forEach((field, i) => curve(field, colors[i], false));
  const cursor = px(states[frameIndex].seconds);
  svgNode(svg, 'line', {x1: cursor, x2: cursor, y1: 15, y2: 212, stroke: '#18353d', 'stroke-dasharray': '3 3'});
}

function render() {
  const sequence = data.cases[caseIndex];
  const state = sequence.states[frameIndex];
  const family = element('model').value;
  element('homevalue').textContent = state.values[family].home_team_epv.toFixed(4);
  element('awayvalue').textContent = state.values[family].away_team_epv.toFixed(4);
  element('homename').textContent = sequence.home;
  element('awayname').textContent = sequence.away;
  element('frame').value = frameIndex;
  element('time').textContent = `${state.seconds.toFixed(1)} s`;
  const gap = frameIndex ? (state.frame-sequence.states[frameIndex-1].frame)/10 : 0;
  element('frameinfo').textContent = `Frame ${state.frame} · Period ${state.period} · Possession: ${state.possession} · ${state.points.filter(p => !p.ball).length}/22 observed players${gap > 2 ? ` · ${gap.toFixed(1)} s gap` : ''}`;
  drawPitch(sequence, state);
  drawTimeline(sequence, family);
}

function stop() {
  if (timer !== null) clearInterval(timer);
  timer = null;
  element('play').textContent = 'Play';
}

function loadCase() {
  stop();
  caseIndex = Number(element('case').value);
  const sequence = data.cases[caseIndex];
  frameIndex = sequence.anchor_position;
  element('frame').max = sequence.states.length-1;
  element('match').textContent = `${sequence.home} – ${sequence.away} · ${sequence.match_id}`;
  element('selection').textContent = sequence.selection;
  render();
}

for (const [i, sequence] of data.cases.entries()) {
  const option = document.createElement('option');
  option.value = i;
  option.textContent = sequence.title;
  element('case').appendChild(option);
}
element('model').value = data.selected_family;
element('case').addEventListener('change', loadCase);
element('model').addEventListener('change', render);
element('targets').addEventListener('change', render);
element('frame').addEventListener('input', () => { stop(); frameIndex = Number(element('frame').value); render(); });
element('play').addEventListener('click', () => {
  if (timer !== null) { stop(); return; }
  if (frameIndex >= data.cases[caseIndex].states.length-1) frameIndex = 0;
  element('play').textContent = 'Pause';
  render();
  timer = setInterval(() => {
    frameIndex++;
    if (frameIndex >= data.cases[caseIndex].states.length-1) stop();
    render();
  }, 650);
});
loadCase();
window.EPV_DEMO_READY = true;
