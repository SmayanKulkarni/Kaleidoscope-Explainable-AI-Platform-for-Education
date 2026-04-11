import { client } from './client';

export const simulate = (body) =>
  client.post('/simulate', body).then((r) => r.data);
