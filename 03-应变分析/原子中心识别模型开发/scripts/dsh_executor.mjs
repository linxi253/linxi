import { DeepSeekHarness } from 'file:///<DSH检出>/packages/sdk/client/lib/index.js';
import { readFile, writeFile, appendFile } from 'node:fs/promises';
const root = '<仓库根>/03-应变分析';
const output = root + '/原子中心识别模型开发/runs/target-adaptation-20260910/';
const route = {provider:'deepseek-official',model:'deepseek-v4.1-flash-expires-on-0910',reasoningEffort:'max'};
const harness = new DeepSeekHarness({dshBin:'<DSH检出>/apps/cli/lib/bin.js',profile:'sdk',
  dshHome:'<用户目录>/.dsh',processCwd:root,cwd:root,...route,initializeTimeoutMs:30000});
const state = {route,workspace:root,workspaceId:'f488056c-261a-48a0-b89e-862ea45f4aa2',status:'starting',startedAt:new Date().toISOString()};
const save=()=>writeFile(output+'dsh_dispatch.json',JSON.stringify(state,null,2));
try {
  await save();
  await harness.start();
  state.status='initialized';await save();
  const prompt = await readFile(root+'/原子中心识别模型开发/reports/DSH_EXECUTION_PLAN.md','utf8');
  const result = await harness.run(prompt,{onNotification(n){
    if(n.params?.sessionId && !state.sessionId){state.sessionId=n.params.sessionId;void save();}
    if(n.method==='session.event'){
      const e=n.params.event;
      if(e?.type==='turn/end'||e?.type==='tool/call'||e?.type==='tool/result'){
        void appendFile(output+'dsh_activity.jsonl',JSON.stringify({time:new Date().toISOString(),type:e.type})+'\n');
      }
    }
  }});
  state.sessionId=result.sessionId;state.status='completed';state.completedAt=new Date().toISOString();
  await writeFile(output+'dsh_final_response.md',result.finalResponse);await save();
  console.log(JSON.stringify(state));
} catch(e){state.status='failed';state.error=String(e);await save();console.error(String(e));process.exitCode=1;}
finally {await harness.close();}
