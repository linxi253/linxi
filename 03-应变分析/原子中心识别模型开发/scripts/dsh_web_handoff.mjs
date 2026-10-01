import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
export const name='atom-target-web-handoff';
export const inject=['sessionController','workspaceRegistry','connection','sessions','agents'];
const root='<仓库根>/03-应变分析/原子中心识别模型开发';
const output=root+'/runs/target-adaptation-20260910/';
await writeFile(output+'dsh_web_module_loaded.txt',new Date().toISOString());
export function apply(ctx){
  void writeFile(output+'dsh_web_apply_started.txt',new Date().toISOString());
  void (async()=>{
    try {
      const existing=await readFile(output+'dsh_web_dispatch.json','utf8').then(JSON.parse).catch(()=>null);
      if(existing?.status==='submitted')return;
      const workspace=await ctx.workspaceRegistry.create('<仓库根>/03-应变分析','03-应变分析');
      const created=await ctx.sessionController.create({workspaceId:workspace.id});
      const sessionId=created.sessionId;
      const route={provider:'deepseek-official',model:'deepseek-v4.1-flash-expires-on-0910',reasoningEffort:'max'};
      const selection=await ctx.sessionController.selectModel({sessionId,...route});
      await ctx.sessionController.rename({sessionId,title:'04、05 原子识别优化 · Codex 执行交接'});
      const plan=await readFile(root+'/reports/DSH_EXECUTION_PLAN.md','utf8');
      const prompt='现在切换到 WebUI 执行，原 SDK 后台已停止。先检查磁盘上已有结果，不要重复训练。请在聊天中简短说明执行进度，完成后等待 Codex 验收。不要创建其他代理。\n\n'+plan;
      await ctx.sessionController.prompt({sessionId,requestId:randomUUID(),mode:'queue',content:[{type:'text',text:prompt}],clientTimeZone:'Asia/Shanghai'},new AbortController().signal);
      const state={status:'submitted',sessionId,workspaceId:workspace.id,workspacePath:workspace.path,
        title:'04、05 原子识别优化 · Codex 执行交接',route:selection.selected,url:'http://127.0.0.1:3081',submittedAt:new Date().toISOString()};
      await writeFile(output+'dsh_web_dispatch.json',JSON.stringify(state,null,2));
      // Local launch URL stays private; this is the official authentication entry.
      await writeFile(output+'.dsh_web_launch_url',ctx.connection.authenticatedUrl('http://127.0.0.1:3081'));
      const agent=ctx.agents.get(sessionId);
      if(agent){await agent.whenIdle();await ctx.sessions.flush(agent.session);
        state.status='idle';state.finishedAt=new Date().toISOString();await writeFile(output+'dsh_web_dispatch.json',JSON.stringify(state,null,2));}
    } catch(e){await writeFile(output+'dsh_web_error.txt',String(e));}
  })();
}
