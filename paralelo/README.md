# Versão paralela (MPI em Go)

## Compilar
```
cd paralelo
go mod init paralelo        # só na primeira vez, se não houver go.mod
go get github.com/mvneves/gompi
go build -o matmul .
```

## Executar
```
mpirun -np 8 ./matmul                      # 1 nó, 8 processos
mpirun -np 32 --oversubscribe ./matmul     # 1 x 32 (mais processos que tarefas do Slurm)
```
Para os layouts multi-nó, use o mesmo padrão do `sequencial.job` / aula 19 (`--nodes`, `--ntasks-per-node`).

Opcional: `-n 1000` muda a dimensão (padrão 5000, como na versão sequencial); útil para testar com P = 1, 3, 7 e comparar a verificação com `./sequencial -n 1000`.

## Saída
Linhas `TEMPO_*` (geração, distribuição, cálculo máx/mín, coleta, total) e uma linha
`RANK r host=... linhas=... calculo=... total=...` por processo.

## Como cada tempo é medido (`mpi.WTime`)
- Todos passam por um `Barrier` antes de `t0`; total = do início da geração ao fim da coleta, máximo entre os processos.
- Geração: só o rank 0, a chamada à função sequencial.
- Distribuição: rank 0 envia o bloco de A de cada processo, B é distribuída em árvore binomial (Send/Recv) e há um `Barrier` no fim, então o tempo vale até todos terem os dados.
- Cálculo: cada processo mede o seu próprio kernel.
- Coleta: rank 0 recebe os blocos de C (inclui espera pelo processo mais lento).
