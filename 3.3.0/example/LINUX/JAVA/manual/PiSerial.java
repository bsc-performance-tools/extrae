public class PiSerial
{
	double m_res;
	long m_n;
	double m_h;

	public PiSerial (long n)
	{
		m_n = n;
		m_h = 1.0 / (double) n;
	}

	public void calculate()
	{
		es.bsc.cepbatools.extrae.Wrapper.Event (1_000, 1);

		m_res = 0;
		for (long i = 1; i <= m_n; i++)
		{
			double x = m_h * ((double)i - 0.5);
			m_res += (4.0 / (1.0 + x*x));
		}

		es.bsc.cepbatools.extrae.Wrapper.Event (1_000, 0);
	}

	public double result()
	{
		return m_res;
	}

}
