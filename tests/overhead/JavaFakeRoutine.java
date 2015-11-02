public class JavaFakeRoutine
{
	long FakeRoutine (long in)
	{
		return in+1;
	}

	public static void main (String [] args)
	{
		long result = 0;
		long niters = 1000000/2;

		JavaFakeRoutine fr = new JavaFakeRoutine();

		long start = System.nanoTime();
		for (long l = 0; l < niters; l++)
		{
			result = result + fr.FakeRoutine(result);
		}
		long end = System.nanoTime();
		System.out.println ("result = " + result);
		System.out.println ("RESULT : es.bsc.cepbatools.extrae.Wrapper.Event() " +
		  (end-start)/niters + " ns");
	}
}

