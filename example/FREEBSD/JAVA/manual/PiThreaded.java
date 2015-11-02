import java.util.*;

public class PiThreaded
{
	long m_n;
	double m_h;
	Vector<PiThread> m_threads;

	public PiThreaded (long n, int nthreads)
	{
		m_n = n;
		m_h = 1.0 / (double) n;

		m_threads = new Vector<PiThread>(nthreads);
		for (long i = 0; i < nthreads; i++)
		{
			m_threads.addElement (
			  new PiThread (m_h,
			    (n/nthreads)*i,
			    (n/nthreads)*(i+1)-1)
			);
		}
	}

	public void calculate()
	{
		es.bsc.cepbatools.extrae.Wrapper.Event (1_000, 2);

		/* Let the threads run */
		for (int i = 0; i < m_threads.size(); i++)
			(m_threads.get(i)).start();

		/* Wait for their work */
		for (int i = 0; i < m_threads.size(); i++)
		{
			try { (m_threads.get(i)).join(); }
			catch (InterruptedException ignore) { }
		}

		es.bsc.cepbatools.extrae.Wrapper.Event (1_000, 0);
	}

	public double result()
	{
		es.bsc.cepbatools.extrae.Wrapper.Event (1_000, 3);

		double res = 0.0;
		for (int i = 0; i < m_threads.size(); i++)
		{
			/* reduce the value to result */
			res += (m_threads.get(i)).result();
		}

		es.bsc.cepbatools.extrae.Wrapper.Event (1_000, 0);

		return res;
	}
}
