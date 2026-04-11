export const loginMock = async (email, password) => {
  // Simulate network
  await new Promise(r => setTimeout(r, 800));
  
  if (email.includes('instructor')) {
    return { id: 'i1', name: 'Dr. Sarah Chen', email, role: 'instructor' };
  } else {
    return { id: 's1', name: 'Alex Johnson', email, role: 'student' };
  }
};

export const signupMock = async (name, email, password, role) => {
  await new Promise(r => setTimeout(r, 800));
  return { id: `u_${Date.now()}`, name, email, role };
};
