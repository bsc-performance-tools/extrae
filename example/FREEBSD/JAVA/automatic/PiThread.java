public class PiThread extends Thread
{
	double m_res;
	double m_h;
	long m_low;
	long m_high;

	public PiThread (double h, long low, long high)
	{
		m_h = h;
		m_low = low;
		m_high = high;
	}

	public void run ()
	{
		m_res = 0.0;
		for (long i = m_low; i <= m_high; i++)
		{
			double x = m_h * ((double)i - 0.5);
			m_res += (4.0 / (1.0 + x*x));
		}
	}

	public double result ()
	{
		return m_res;
	}
}
