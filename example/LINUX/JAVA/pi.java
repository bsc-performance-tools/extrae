public class pi
{
	private static final int pi_type[] = { 1000 };
	private static final long entry_value[] = { 1 };
	private static final long exit_value[] = { 0 };
	private double h;
	private long n;

	public pi (long N)
	{
		n = N;
		h = 1.0 / (double) n;
	}

	public double calculate()
	{
		double tmp = 0;

		es.bsc.cepbatools.extrae.Wrapper.nEvent (pi_type, entry_value);

		for (long i = 1; i <= n; i++)
		{
			double x = h * ((double)i - 0.5);
			tmp += (4.0 / (1.0 + x*x));
		}

		es.bsc.cepbatools.extrae.Wrapper.nEvent (pi_type, exit_value);

		return tmp / (double) n;
	}

	public long steps()
	{
		return n;
	}

	public static void main(String args[])
	{
		// es.bsc.cepbatools.extrae.Wrapper.Init();

		pi p = new pi (100000);
		System.out.println ("PI calculated with " + p.steps() +
		  " steps = " + p.calculate());

		int pi_prv_type = pi_type[0];
		String pi_prv_type_description = "PI::calculate()";
		long pi_prv_values[] = { entry_value[0], exit_value[0] };
		String pi_prv_values_descriptions [] = { "Begin", "End" };
		es.bsc.cepbatools.extrae.Wrapper.defineEventType (pi_prv_type,
		  pi_prv_type_description, pi_prv_values,
		  pi_prv_values_descriptions);

		// es.bsc.cepbatools.extrae.Wrapper.Fini();
	}
}

