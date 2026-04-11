export const getStudentData = async () => {
  await new Promise(r => setTimeout(r, 600));
  return {
    student: { name: 'Alex Johnson', title: 'AI Scholar', avatarInitials: 'AJ' },
    riskScore: 78,
    riskLevel: 'HIGH',
    trustScore: 0.92,
    trustLevel: 'HIGH',
    topAction: { label: 'Complete Module 4 Quiz', riskReduction: 12 },
    shapFeatures: [
      { name: 'Module Engagement', value: 0.65, direction: 'increases' },
      { name: 'Quiz Scores',       value: 0.40, direction: 'increases' },
      { name: 'Forum Participation', value: 0.30, direction: 'decreases' },
      { name: 'Assignment Completion', value: 0.22, direction: 'decreases' },
      { name: 'Video Watch %',     value: 0.15, direction: 'increases' },
    ],
    featureContributions: [
      { name: 'Days since last login', value: 0.8, direction: 'increases' },
      { name: 'Late submission count', value: 0.5, direction: 'increases' },
      { name: 'Peer review score', value: 0.4, direction: 'decreases' },
    ],
    conceptAnalysis: [
      { name: 'Gradients & Backprop', value: 0.7, direction: 'increases' },
      { name: 'Data Loaders', value: 0.3, direction: 'increases' },
      { name: 'Linear Algebra', value: 0.5, direction: 'decreases' },
    ],
    lookalikes: [
      { id: '842', match: 95, outcome: 'DROPPED OUT' },
      { id: '311', match: 92, outcome: 'COMPLETED' },
      { id: '559', match: 88, outcome: 'DROPPED OUT' },
    ],
    aiNarrative: "Alex, the data suggests you've encountered a bit of a hurdle in Module 4. If you spend just 30 minutes on the Module 4 quiz this evening, your risk score could drop by nearly 15%. You're closer to completion than it feels right now!",
  };
};

export const getBaselineFeatures = async () => {
  await new Promise(r => setTimeout(r, 400));
  return [
    { id: 'study_hours',    label: 'Weekly Study Hours',       value: 15, min: 0, max: 60,  unit: 'hrs',  weight: 0.45, direction: 'reduces' },
    { id: 'quiz_avg',       label: 'Quiz Average Score',       value: 74, min: 0, max: 100, unit: '%',    weight: 0.38, direction: 'reduces' },
    { id: 'peer_freq',      label: 'Peer Interaction Freq.',   value: 4,  min: 0, max: 10,  unit: '/10',  weight: 0.20, direction: 'reduces' },
    { id: 'forum_posts',    label: 'Forum Posts per Week',     value: 2,  min: 0, max: 50,  unit: 'posts',weight: 0.30, direction: 'reduces' },
    { id: 'completion',     label: 'Assignment Completion',    value: 72, min: 0, max: 100, unit: '%',    weight: 0.35, direction: 'reduces' },
  ];
};

export const simulateWhatIf = async (sliders) => {
  await new Promise(r => setTimeout(r, 60)); // mimic FastSHAP latency
  const baseline = {'Assignments Completed': 8, 'Days Since Last Login': 12, 'Forum Posts': 2};
  const BASE_RISK = 14; 
  let delta = 0;
  
  if (sliders['Assignments Completed'] !== undefined) {
      delta -= (sliders['Assignments Completed'] - baseline['Assignments Completed']) * 0.5;
  }
  if (sliders['Days Since Last Login'] !== undefined) {
      delta += (sliders['Days Since Last Login'] - baseline['Days Since Last Login']) * 1.2;
  }
  if (sliders['Forum Posts'] !== undefined) {
      delta -= (sliders['Forum Posts'] - baseline['Forum Posts']) * 0.8;
  }
  
  const newRisk = Math.max(0, Math.min(100, Math.round(BASE_RISK + delta)));
  
  const impacts = [];
  if (sliders['Assignments Completed'] !== undefined) {
      impacts.push({ name: 'Assignments Completed', impact: (sliders['Assignments Completed'] - baseline['Assignments Completed']) * 0.5 });
  }
  if (sliders['Days Since Last Login'] !== undefined) {
      impacts.push({ name: 'Days Since Last Login', impact: -(sliders['Days Since Last Login'] - baseline['Days Since Last Login']) * 1.2 });
  }
  if (sliders['Forum Posts'] !== undefined) {
      impacts.push({ name: 'Forum Posts', impact: (sliders['Forum Posts'] - baseline['Forum Posts']) * 0.8 });
  }

  return {
    newRisk: newRisk,
    impactFactors: impacts
  };
};

export const getActionPlan = async () => {
  await new Promise(r => setTimeout(r, 500));
  return {
    actions: [
      { id: 'ap1', priority: 'P1', type: 'CAUSAL', feature: 'Engagement', current: '15%', target: '25%', recommendation: 'Increase forum participation.', riskReduction: 18 },
      { id: 'ap2', priority: 'P2', type: 'CAUSAL', feature: 'Assessment', current: '62%', target: '70%', recommendation: 'Attend office hours.', riskReduction: 10 },
      { id: 'ap3', priority: 'P3', type: 'CORRELATED', feature: 'Reading', current: 'Low', target: 'Medium', recommendation: 'Review materials.', riskReduction: 5 },
    ],
    combinedRiskReduction: 33,
    finalSuccessProbability: 92,
  };
};

export const getInstructorData = async () => {
  await new Promise(r => setTimeout(r, 600));
  return {
    classStats: { riskScore: 14.2, modelConfidence: 0.89, stabilityScore: 0.94, fidelityScore: 0.91 },
    shapFeatures: [
      { name: 'Attendance', value: 0.45, direction: 'positive', type: 'causal' },
      { name: 'Late Subs', value: -0.28, direction: 'negative', type: 'correlated' },
    ],
    anchorRule: 'IF Attendance < 75% THEN Risk Level = HIGH.',
    modelComparison: { modelA: { name: 'XGBoost', auc: 0.89 }, modelB: { name: 'RF', auc: 0.88 }, agreement: 92 },
    trustMetrics: { fidelity: 91, stability: 94, completeness: 88 },
    interventions: [
      { feature: 'Course Attendance', current: '62.4%', target: '75%', impact: '+12.4%', type: 'Causal', priority: 'Critical' },
    ],
  };
};
