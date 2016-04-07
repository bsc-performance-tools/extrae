public class PiExample
{
	public static void main (String [] args)
	{
		long steps = 100_000_000;
		long begin, end;

		PiSerial pis = new PiSerial (steps);
		begin = System.currentTimeMillis();
		pis.calculate ();
		end = System.currentTimeMillis();
		System.out.println ("Serial PI (" + steps + " steps) = " + pis.result()
		  + " took " + (end - begin) + " ms");

		PiThreaded pit1 = new PiThreaded (steps, 1);
		begin = System.currentTimeMillis();
		pit1.calculate ();
		end = System.currentTimeMillis();
		System.out.println ("Threaded PI (" + steps + " steps, 1 thread)  = " + pit1.result()
		  + " took " + (end - begin) + " ms");

		PiThreaded pit2 = new PiThreaded (steps, 2);
		begin = System.currentTimeMillis();
		pit2.calculate ();
		end = System.currentTimeMillis();
		System.out.println ("Threaded PI (" + steps + " steps, 2 threads) = " + pit2.result()
		  + " took " + (end - begin) + " ms");

		PiThreaded pit4 = new PiThreaded (steps, 4);
		begin = System.currentTimeMillis();
		pit4.calculate ();
		end = System.currentTimeMillis();
		System.out.println ("Threaded PI (" + steps + " steps, 4 threads) = " + pit4.result()
		  + " took " + (end - begin) + " ms");

		PiThreaded pit8 = new PiThreaded (steps, 8);
		begin = System.currentTimeMillis();
		pit8.calculate ();
		end = System.currentTimeMillis();
		System.out.println ("Threaded PI (" + steps + " steps, 8 threads) = " + pit8.result()
		  + " took " + (end - begin) + " ms");
	}
}
