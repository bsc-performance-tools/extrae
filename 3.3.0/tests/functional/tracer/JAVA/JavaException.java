class JavaException {

	JavaException () throws Exception
	{
		throw new Exception ("Exception");
	}

	public static void main (String args[])
	{
		try
		{ JavaException js = new JavaException(); }
		catch (Exception e)
		{ }
	}
}
