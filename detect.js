/* ── Pixel Sampling Helpers ── */
function sampleGrid(ctx, cx, cy, radius) {
    const colors = [];
    const w = ctx.canvas.width, h = ctx.canvas.height;
    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            const x = Math.max(0, Math.min(Math.round(cx + dx), w - 1));
            const y = Math.max(0, Math.min(Math.round(cy + dy), h - 1));
            const d = ctx.getImageData(x, y, 1, 1).data;
            colors.push({ r: d[0], g: d[1], b: d[2] });
        }
    }
    return colors;
}

function medianColor(colors) {
    if (!colors.length) return { r: 180, g: 150, b: 130 };
    const sorted = [...colors].sort((a, b) => (a.r + a.g + a.b) - (b.r + b.g + b.b));
    return sorted[Math.floor(sorted.length / 2)];
}

function brightnessOf(c) { return (c.r + c.g + c.b) / 3; }

function avgBrightnessOfPts(ctx, pts) {
    if (!pts.length) return 128;
    let total = 0;
    const w = ctx.canvas.width, h = ctx.canvas.height;
    for (const p of pts) {
        const x = Math.max(0, Math.min(Math.round(p.x), w - 1));
        const y = Math.max(0, Math.min(Math.round(p.y), h - 1));
        const d = ctx.getImageData(x, y, 1, 1).data;
        total += (d[0] + d[1] + d[2]) / 3;
    }
    return total / pts.length;
}

function avgColorOfPts(ctx, pts) {
    if (!pts.length) return { r: 60, g: 60, b: 60 };
    let r = 0, g = 0, b = 0;
    const w = ctx.canvas.width, h = ctx.canvas.height;
    for (const p of pts) {
        const x = Math.max(0, Math.min(Math.round(p.x), w - 1));
        const y = Math.max(0, Math.min(Math.round(p.y), h - 1));
        const d = ctx.getImageData(x, y, 1, 1).data;
        r += d[0]; g += d[1]; b += d[2];
    }
    const n = pts.length;
    return { r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n) };
}

function clampBrightness(c, lo, hi) {
    const br = brightnessOf(c);
    if (br < lo) { const f = lo / Math.max(br, 1); return mapC(c, f); }
    if (br > hi) { const f = hi / br; return mapC(c, f); }
    return c;
}

function mapC(c, f) {
    return {
        r: Math.min(255, Math.round(c.r * f)),
        g: Math.min(255, Math.round(c.g * f)),
        b: Math.min(255, Math.round(c.b * f))
    };
}

/* ── Color Helpers ── */
function rgbHex(c) {
    return '#' + [c.r, c.g, c.b].map(v => v.toString(16).padStart(2, '0')).join('');
}
function brighten(hex, f) {
    const c = hexRgb(hex);
    return rgbHex(mapC(c, f));
}
function hexRgb(h) {
    return { r: parseInt(h.slice(1,3),16), g: parseInt(h.slice(3,5),16), b: parseInt(h.slice(5,7),16) };
}

/* ── Feature Extraction ── */
function extractFeatures(det, ctx) {
    const lm = det.landmarks;
    const age = Math.round(det.age);
    const gender = det.gender;
    const genderProb = det.genderProbability;

    const exprs = det.expressions;
    const sortedExprs = Object.entries(exprs).sort((a, b) => b[1] - a[1]);
    const expression = sortedExprs[0][0];
    const expressionScore = sortedExprs[0][1];

    const jaw = lm.getJawOutline();
    const nose = lm.getNose();
    const mouth = lm.getMouth();
    const leftBrow = lm.getLeftEyeBrow();
    const rightBrow = lm.getRightEyeBrow();
    const leftEye = lm.getLeftEye();
    const rightEye = lm.getRightEye();

    // Skin color
    const skinSamples = [];
    skinSamples.push(...sampleGrid(ctx, (leftBrow[2].x + rightBrow[2].x) / 2, (leftBrow[2].y + rightBrow[2].y) / 2, 3));
    skinSamples.push(...sampleGrid(ctx, nose[0].x - 8, nose[0].y, 2));
    skinSamples.push(...sampleGrid(ctx, nose[0].x + 8, nose[0].y, 2));

    let skinColor = medianColor(skinSamples);
    skinColor = mapC(skinColor, 1.15);
    skinColor = clampBrightness(skinColor, 95, 245);
    const skinBr = brightnessOf(skinColor);

    // Hair color
    const allHairSamples = [];
    const brows = [...leftBrow, ...rightBrow];
    const faceH = jaw[8].y - Math.min(...brows.map(p => p.y));
    const hairOffset = Math.max(35, faceH * 0.65);
    for (const offsetMul of [0.8, 1.0, 1.2]) {
        for (const p of brows) {
            const y = p.y - hairOffset * offsetMul;
            if (y > 2) allHairSamples.push(...sampleGrid(ctx, p.x, y, 2));
        }
    }
    let hairColor = allHairSamples.length > 0 ? medianColor(allHairSamples) : { r: 60, g: 40, b: 30 };
    hairColor = clampBrightness(hairColor, 15, 210);

    // Expression
    let refinedExpression = 'Normal';
    if (expression === 'happy') refinedExpression = expressionScore > 0.85 ? 'Überglücklich' : 'Normal lachend';
    else if (expression === 'sad') refinedExpression = expressionScore > 0.7 ? 'Voll traurig' : 'Traurig';
    else refinedExpression = expression;

    // Mustache
    const mustacheZone = [];
    for (let t = 0.2; t <= 0.8; t += 0.3) {
        mustacheZone.push(...sampleGrid(ctx, nose[6].x, nose[6].y + (mouth[3].y - nose[6].y) * t, 2));
    }
    const hasMustache = avgBrightnessOfPts(ctx, mustacheZone) < skinBr * 0.78;

    // Teeth
    let showsTeeth = false;
    if (expression === 'happy') {
        const mCenter = { x: (mouth[12].x + mouth[16].x) / 2, y: (mouth[12].y + mouth[16].y) / 2 };
        showsTeeth = avgBrightnessOfPts(ctx, sampleGrid(ctx, mCenter.x, mCenter.y, 2)) > skinBr * 0.8;
    }

    // Nose Shape
    const noseShape = (Math.abs(nose[4].x - nose[0].x) / (Math.abs(nose[3].y - nose[0].y) || 1)) > 1.3 ? 'breit' : 'schmal';

    // Earrings
    const earringPts = [];
    earringPts.push(...sampleGrid(ctx, jaw[0].x - 5, jaw[0].y + 5, 2));
    earringPts.push(...sampleGrid(ctx, jaw[16].x + 5, jaw[16].y + 5, 2));
    const hasEarrings = Math.abs(avgBrightnessOfPts(ctx, earringPts) - skinBr) > 40;

    // Glasses
    const glasses = detectGlasses(ctx, lm, skinBr);

    return {
        age, gender, genderProb, expression, refinedExpression,
        skinColor, hairColor, skinHex: rgbHex(skinColor), hairHex: rgbHex(hairColor),
        glasses, hasMustache, showsTeeth, noseShape, hasEarrings
    };
}

function detectGlasses(ctx, landmarks, skinBr) {
    const le = landmarks.getLeftEye(), re = landmarks.getRightEye();
    const lb = landmarks.getLeftEyeBrow(), rb = landmarks.getRightEyeBrow();
    function box(pts) {
        let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
        for (const p of pts) { x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y); x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y); }
        return { x1, y1, x2, y2, cx: (x1+x2)/2, cy: (y1+y2)/2, w: x2-x1, h: y2-y1 };
    }
    const lBox = box(le), rBox = box(re), lbBox = box(lb), rbBox = box(rb);
    const topPts = [];
    for (const [eB, bB] of [[lBox, lbBox], [rBox, rbBox]]) {
        const y = (bB.y2 + eB.y1) / 2;
        for (let x = eB.x1 - 4; x <= eB.x2 + 4; x += 2) topPts.push({ x, y }, { x, y: y + 2 });
    }
    const bridgePts = [];
    for (let t = 0.1; t <= 0.9; t += 0.1) bridgePts.push({ x: le[3].x + (re[0].x - le[3].x) * t, y: le[3].y + (re[0].y - le[3].y) * t - 2 });
    const hasGlasses = (avgBrightnessOfPts(ctx, topPts) / Math.max(skinBr, 1) < 0.72 || avgBrightnessOfPts(ctx, bridgePts) / Math.max(skinBr, 1) < 0.72);
    if (!hasGlasses) return { hasGlasses: false };
    const glassW = (lBox.w + rBox.w) / 2 + 10, glassH = (lBox.h + rBox.h) / 2 + 10;
    const ar = glassW / Math.max(glassH, 1);
    return { hasGlasses: true, shape: ar > 1.6 ? 'eckig' : ar < 1.2 ? 'rund' : 'oval', colorHex: rgbHex(avgColorOfPts(ctx, bridgePts)) };
}
