using System;
namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class WalkerIdentityTests
{
    [Fact]
    public void DenseUniformMatchesWalkerOnZeroOne()
    {
        const ulong key = 0xA5A5_1234UL | (0x9E37UL << 32);
        var table = RngDenseUniformOracle.Build(0f, 1f);
        int mismatch = 0, deep = 0; string first = "";
        for (long i = 0; i < 200000; i++)
        {
            ulong draw = RngTestOracle.DrawValue(key, 0, i);
            float dense = BitConverter.UInt32BitsToSingle(RngDenseUniformOracle.SampleBits(table, draw));
            float walker = RngTestOracle.DrawUniform(key, 0, i);
            if (draw >> 23 < 8) deep++;
            if (BitConverter.SingleToUInt32Bits(dense) != BitConverter.SingleToUInt32Bits(walker))
            {
                if (mismatch++ == 0)
                    first = $" first i={i} dense={dense:G9} walker={walker:G9} selector={draw >> 23}";
            }
        }
        System.IO.File.WriteAllText("/tmp/claude-0/-home-user/ef825434-0b5b-5603-a3da-e37e26468e07/scratchpad/walker.txt",
            $"mismatches={mismatch}/200000 deepDraws={deep}{first}\n");
        Assert.Equal(0, mismatch);
    }
}
