public class JavaEvent
{
	public static void main (String [] args)
	{
		long niters = 1000000;

		long start = System.nanoTime();
		for (long l = 0; l < niters; l++)
		{
			es.bsc.cepbatools.extrae.Wrapper.Event (1, 1+l);
		}
		long end = System.nanoTime();
		System.out.println ("RESULT : es.bsc.cepbatools.extrae.Wrapper.Event() " +
		  (end-start)/niters + " ns");
	}
}
