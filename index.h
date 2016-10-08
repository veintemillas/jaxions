static inline void idx2Vec(int idx, int x[3], const int n1)
{
	x[0] = idx%n1;
	x[2] = idx/(n1*n1);
	x[1] = (idx - x[2]*(n1*n1))/n1;
}

static inline int vec2Idx(int x[3], const int n1)
{
	return (x[0] + x[1]*n1 + x[2]*(n1*n1));
}
