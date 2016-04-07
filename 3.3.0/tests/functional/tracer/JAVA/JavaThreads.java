class JavaThreads extends Thread
{
	JavaThreads ()
	{ }

	public void run ()
	{ try { sleep (1); } catch (Exception e) { } }

	public static void main (String args[])
	{
		if (args.length > 0)
		{
			int nthreads = Integer.parseInt (args[0]);
			JavaThreads t[] = new JavaThreads[nthreads];
			for (int i = 0; i < nthreads; i++)
			{
				t[i] = new JavaThreads();
				t[i].start();
			}
			for (int i = 0; i < nthreads; i++)
				try { t[i].join(); } catch (Exception e) { }
		}
	}
}
