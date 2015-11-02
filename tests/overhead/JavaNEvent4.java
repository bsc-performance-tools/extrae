public class JavaNEvent4
{
	public static void main (String [] args)
	{
		long niters = 1000000;

		long values[] = { 1, 2, 3, 4 };
		int types[] = { 4, 3, 2, 1 };

		long start = System.nanoTime();
		for (long l = 0; l < niters; l++)
		{
			es.bsc.cepbatools.extrae.Wrapper.nEvent (types, values);
		}
		long end = System.nanoTime();
		System.out.println ("RESULT : es.bsc.cepbatools.extrae.Wrapper.nEvent(<4>) " +
		  (end-start)/niters + " ns");
	}
}

