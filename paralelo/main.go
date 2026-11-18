// T2 FPPD - Multiplicação de matrizes C = A x B com MPI (Go + gompi).
//
// Divisão do trabalho: decomposição por blocos de linhas de A.
//   - O rank 0 gera A e B (função sequencial, sem alteração).
//   - Cada processo p recebe o bloco de linhas [r0, r1) de A, e todos recebem B inteira.
//   - Cada processo calcula as linhas [r0, r1) de C (kernel sequencial, sem alteração).
//   - O rank 0 coleta os blocos de C e chama a verificação sequencial.
//
// Como N pode não ser divisível pelo número de processos P, as N%P primeiras
// ranks recebem uma linha a mais (diferença máxima de 1 linha entre processos).
package main

import (
	"flag"
	"fmt"
	"os"

	mpi "github.com/mvneves/gompi"
	"github.com/mvneves/gompi/comm"
)

// N é a dimensão das matrizes (flag -n, padrão 5000, como na versão sequencial).
var N = 5000

const (
	tagA     = 1 // bloco de linhas de A (rank 0 -> demais)
	tagB     = 2 // matriz B (broadcast em árvore)
	tagC     = 3 // bloco de linhas de C (demais -> rank 0)
	tagTempo = 4 // tempos de cada processo (demais -> rank 0)
	tagHost  = 5 // hostname de cada processo (demais -> rank 0)
)

// faixa devolve o intervalo de linhas [r0, r1) do processo "rank".
func faixa(rank, size, n int) (r0, r1 int) {
	base := n / size
	rem := n % size
	if rank < rem {
		r0 = rank * (base + 1)
		r1 = r0 + base + 1
	} else {
		r0 = rem*(base+1) + (rank-rem)*base
		r1 = r0 + base
	}
	return
}

// bcastArvore distribui buf do rank 0 para todos em ceil(log2(P)) etapas
// (árvore binomial) usando apenas Send/Recv bloqueantes.
func bcastArvore(c comm.Communicator, buf []float64, rank, size int) {
	for mask := 1; mask < size; mask <<= 1 {
		if rank < mask {
			if dest := rank + mask; dest < size {
				c.Send(buf, dest, tagB)
			}
		} else if rank < 2*mask {
			c.Recv(buf, rank-mask, tagB)
		}
	}
}

func main() {
	flag.IntVar(&N, "n", 5000, "dimensão das matrizes")
	flag.Parse()

	mpi.Init()
	defer mpi.Finalize()

	world := mpi.NewComm(true)
	rank := world.GetRank()
	size := world.GetSize()

	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}

	r0, r1 := faixa(rank, size, N)
	linhas := r1 - r0

	// Todos começam juntos; t0 marca o início da geração (rank 0).
	world.Barrier()
	t0 := mpi.WTime()

	// ---------------------------------------------------------------- geração
	var A, B, C []float64 // A, C: completas só no rank 0
	var aLocal, cLocal []float64
	var tGeracao float64

	if rank == 0 {
		tg := mpi.WTime()
		A, B = geraMatrizes(N) // função sequencial, sem alteração
		tGeracao = mpi.WTime() - tg
		C = make([]float64, N*N)
		aLocal = A[r0*N : r1*N]
		cLocal = C[r0*N : r1*N]
	} else {
		B = make([]float64, N*N)
		aLocal = make([]float64, linhas*N)
		cLocal = make([]float64, linhas*N)
	}

	// ------------------------------------------------------------ distribuição
	// Rank 0 envia o bloco de A de cada processo; depois B vai em árvore.
	// O Barrier ao final faz o tempo de distribuição (medido no rank 0)
	// valer até o momento em que TODOS os processos já têm os seus dados.
	td := mpi.WTime()
	if rank == 0 {
		for r := 1; r < size; r++ {
			a0, a1 := faixa(r, size, N)
			if a1 > a0 {
				world.Send(A[a0*N:a1*N], r, tagA)
			}
		}
	} else if linhas > 0 {
		world.Recv(aLocal, 0, tagA)
	}
	bcastArvore(world, B, rank, size)
	world.Barrier()
	tDistribuicao := mpi.WTime() - td

	// ----------------------------------------------------------------- cálculo
	tc := mpi.WTime()
	if linhas > 0 {
		multiplica(aLocal, B, cLocal, N) // kernel sequencial, sem alteração
	}
	tCalculo := mpi.WTime() - tc

	// ------------------------------------------------------------------- coleta
	if rank != 0 {
		if linhas > 0 {
			world.Send(cLocal, 0, tagC)
		}
		tTotal := mpi.WTime() - t0
		world.Send([]float64{tCalculo, tTotal}, 0, tagTempo)
		world.Send(host, 0, tagHost)
		return
	}

	tcol := mpi.WTime()
	for r := 1; r < size; r++ {
		c0, c1 := faixa(r, size, N)
		if c1 > c0 {
			world.Recv(C[c0*N:c1*N], r, tagC)
		}
	}
	tColeta := mpi.WTime() - tcol
	tTotalRank0 := mpi.WTime() - t0

	// Tempos e hostnames dos demais processos (fora das medições de fase).
	calculo := make([]float64, size)
	total := make([]float64, size)
	hosts := make([]string, size)
	calculo[0], total[0], hosts[0] = tCalculo, tTotalRank0, host
	for r := 1; r < size; r++ {
		tm := make([]float64, 2)
		world.Recv(tm, r, tagTempo)
		hs, _ := world.RecvString(r, tagHost)
		calculo[r], total[r], hosts[r] = tm[0], tm[1], hs
	}

	// Tempo total = o do processo mais lento.
	tTotal, calcMin, calcMax := total[0], calculo[0], calculo[0]
	for r := 1; r < size; r++ {
		if total[r] > tTotal {
			tTotal = total[r]
		}
		if calculo[r] < calcMin {
			calcMin = calculo[r]
		}
		if calculo[r] > calcMax {
			calcMax = calculo[r]
		}
	}

	// ----------------------------------------------------------------- saída
	fmt.Printf("N=%d processos=%d\n", N, size)
	verifica(C, N) // verificação sequencial, sem alteração: mesmos valores da versão sequencial
	fmt.Printf("TEMPO_GERACAO      %.6f\n", tGeracao)
	fmt.Printf("TEMPO_DISTRIBUICAO %.6f\n", tDistribuicao)
	fmt.Printf("TEMPO_CALCULO_MAX  %.6f\n", calcMax)
	fmt.Printf("TEMPO_CALCULO_MIN  %.6f\n", calcMin)
	fmt.Printf("TEMPO_COLETA       %.6f\n", tColeta)
	fmt.Printf("TEMPO_TOTAL        %.6f\n", tTotal)
	for r := 0; r < size; r++ {
		a, b := faixa(r, size, N)
		fmt.Printf("RANK %d host=%s linhas=%d calculo=%.6f total=%.6f\n",
			r, hosts[r], b-a, calculo[r], total[r])
	}
}
