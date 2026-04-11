/**
 * Builds the 34-feature InstructorRecoItem required by POST /recommend/instructor.
 *
 * @param {string} studentId
 * @param {string} learnerId
 * @param {{ archetype: string, department: string, teaching_style: string,
 *            experience_years: number, avg_cohort_size: number,
 *            intervention_intensity: number, past_success_rate: number,
 *            confidence_score: number }} instructor
 * @param {{ current_module: string, current_presentation: string,
 *            dropout_risk_score: number, risk_trajectory: string,
 *            learning_momentum: number, quiz_avg_score: number,
 *            assignment_submission_rate: number, missed_deadlines: number,
 *            days_inactive: number, week_in_course: number,
 *            vs_cohort_quiz_delta: number, peer_collab_readiness: number }} student
 * @param {{ recommended_module: string, recommended_presentation: string,
 *            affinity_score: number, intervention_type: string,
 *            intervention_urgency: string, content_type: string,
 *            effort_hours: number }} target
 * @param {{ signal_score: number, avg_quiz_score: number, avg_dropout_rate: number,
 *            size: number, engagement_percentile: number }} cohort
 * @returns {{ item_id: string, features: object }}
 */
export function buildInstructorRecoItem(studentId, learnerId, instructor, student, target, cohort) {
  return {
    item_id: `${learnerId}-${target.recommended_module}`,
    features: {
      instructor_archetype:               instructor.archetype,
      instructor_department:              instructor.department,
      instructor_teaching_style:          instructor.teaching_style,
      instructor_experience_years:        instructor.experience_years,
      instructor_avg_cohort_size:         instructor.avg_cohort_size,
      instructor_intervention_intensity:  instructor.intervention_intensity,
      instructor_past_success_rate:       instructor.past_success_rate,
      instructor_confidence_score:        instructor.confidence_score,
      student_id:                         parseInt(studentId, 10) || 0,
      learner_id:                         learnerId,
      student_current_module:             student.current_module,
      student_current_presentation:       student.current_presentation,
      student_dropout_risk_score:         student.dropout_risk_score,
      student_risk_trajectory:            student.risk_trajectory,
      student_learning_momentum:          student.learning_momentum,
      student_quiz_avg_score:             student.quiz_avg_score,
      student_assignment_submission_rate: student.assignment_submission_rate,
      student_missed_deadlines:           student.missed_deadlines,
      student_days_inactive:              student.days_inactive,
      student_week_in_course:             student.week_in_course,
      student_vs_cohort_quiz_delta:       student.vs_cohort_quiz_delta,
      student_peer_collab_readiness:      student.peer_collab_readiness,
      recommended_module:                 target.recommended_module,
      recommended_presentation:           target.recommended_presentation,
      student_affinity_score:             target.affinity_score,
      cohort_signal_score:                cohort.signal_score,
      cohort_avg_quiz_score:              cohort.avg_quiz_score,
      cohort_avg_dropout_rate:            cohort.avg_dropout_rate,
      cohort_size:                        cohort.size,
      cohort_engagement_percentile:       cohort.engagement_percentile,
      intervention_type:                  target.intervention_type,
      intervention_urgency:               target.intervention_urgency,
      recommended_content_type:           target.content_type,
      estimated_effort_hours:             target.effort_hours,
    },
  };
}
