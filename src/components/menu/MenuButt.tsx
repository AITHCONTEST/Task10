import img from "../../assets/new/menu.svg";
import Button from "../Button";
import { useState } from "react";
import Menu from "./Menu";

const MenuButt = () => {
    const [isActive, setIsActive] = useState(false);

    const dynamicStyle = {
        transform: isActive ? 'rotateZ(180deg)' : 'rotateZ(0deg)',
        transition: 'transform 0.3s ease',
        height: '24px',
        zIndex: 110,
        
    };

    const MenuDynamicStyle = {
        transform: isActive ? 'translateX(0)' : 'translateX(-100%)',
    };

    const tap = ()=>{
        setIsActive(!isActive);
    }    
    return(
        <>
            <div style={dynamicStyle}>
                <Button img={img} tap={tap}  alt={"menu"}/>
                
            </div>
            <Menu tap={()=>{setIsActive(!isActive)}} style={MenuDynamicStyle}/>
        </>
        
    );
};

export default MenuButt;