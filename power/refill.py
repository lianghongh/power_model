import re

def refill_stats(data_path,template_path,save_path):
    with open(template_path,"r",encoding="utf-8") as f:
        template=f.read()
    with open(data_path,"r",encoding="utf-8") as f:
        with open(save_path,"w+",encoding="utf-8") as fw:
            line=f.readline()
            while line!="\n" and line!="":
                print(line,end="")
                line=line.split()
                template=re.sub("sim_insts +\\d+","sim_insts                                  "+line[0],template)
                template=re.sub("system.cpu0.Branches +\\d+","system.cpu0.Branches                    "+str(int(line[1])//4),template)
                template=re.sub("system.cpu1.Branches +\\d+","system.cpu1.Branches                    "+str(int(line[1])//4),template)
                template=re.sub("system.cpu2.Branches +\\d+","system.cpu2.Branches                    "+str(int(line[1])//4),template)
                template=re.sub("system.cpu3.Branches +\\d+","system.cpu3.Branches                    "+str(int(line[1])//4),template)
                template=re.sub("system.l2.overall_accesses::total +\\d+","system.l2.overall_accesses::total         "+line[2],template)
                template=re.sub("system.cpu0.icache.overall_accesses::total +\\d+","system.cpu0.icache.overall_accesses::total   "+str(int(line[3])//4),template)
                template=re.sub("system.cpu1.icache.overall_accesses::total +\\d+","system.cpu1.icache.overall_accesses::total   "+str(int(line[3])//4),template)
                template=re.sub("system.cpu2.icache.overall_accesses::total +\\d+","system.cpu2.icache.overall_accesses::total   "+str(int(line[3])//4),template)
                template=re.sub("system.cpu3.icache.overall_accesses::total +\\d+","system.cpu3.icache.overall_accesses::total   "+str(int(line[3])//4),template)
                template=re.sub("system.cpu0.dcache.overall_accesses::total +\\d+","system.cpu0.dcache.overall_accesses::total   "+str(int(line[4])//4),template)
                template=re.sub("system.cpu1.dcache.overall_accesses::total +\\d+","system.cpu1.dcache.overall_accesses::total   "+str(int(line[4])//4),template)
                template=re.sub("system.cpu2.dcache.overall_accesses::total +\\d+","system.cpu2.dcache.overall_accesses::total   "+str(int(line[4])//4),template)
                template=re.sub("system.cpu3.dcache.overall_accesses::total +\\d+","system.cpu3.dcache.overall_accesses::total   "+str(int(line[4])//4),template)
                template=re.sub("system.cpu0.num_mem_refs +\\d+","system.cpu0.num_mem_refs         "+str(int(line[5])//4),template)
                template=re.sub("system.cpu1.num_mem_refs +\\d+","system.cpu1.num_mem_refs         "+str(int(line[5])//4),template)
                template=re.sub("system.cpu2.num_mem_refs +\\d+","system.cpu2.num_mem_refs         "+str(int(line[5])//4),template)
                template=re.sub("system.cpu3.num_mem_refs +\\d+","system.cpu3.num_mem_refs         "+str(int(line[5])//4),template)
                template=re.sub("system.l2.writebacks::total +\\d+","system.l2.writebacks::total              "+line[6],template)
                template=re.sub("system.cpu0.numCycles +\\d+","system.cpu0.numCycles                       "+str(int(line[7])//4),template)
                template=re.sub("system.cpu1.numCycles +\\d+","system.cpu1.numCycles                       "+str(int(line[7])//4),template)
                template=re.sub("system.cpu2.numCycles +\\d+","system.cpu2.numCycles                       "+str(int(line[7])//4),template)
                template=re.sub("system.cpu3.numCycles +\\d+","system.cpu3.numCycles                       "+str(int(line[7])//4),template)
                fw.write(template)
                line=f.readline()


if __name__=="__main__":
    template_path="/Users/lianghong/Desktop/GraduateData/research1/run/power_test/stats/stats_100.txt"
    data_path="/Users/lianghong/Desktop/GraduateData/research1/data/blackscholes/data/cov"
    save_path="/Users/lianghong/Desktop/mystats.txt"
    refill_stats(data_path,template_path,save_path)